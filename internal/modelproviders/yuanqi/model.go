package yuanqi

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

const (
	defaultBaseURL        = "https://yuanqi.tencent.com/openapi/v1/agent/chat/completions"
	defaultTimeout        = 30 * time.Second
	defaultMaxConcurrency = 10
	defaultMaxRetries     = 2
	defaultRetryDelay     = 200 * time.Millisecond
	maxRetryDelay         = 5 * time.Second
	maxErrorBodyBytes     = 4096
)

// Config is the configuration for Yuanqi ChatModel.
type Config struct {
	BaseURL string

	AppKey      string
	AssistantID string
	UserID      string

	Timeout time.Duration
	// MaxConcurrency caps concurrent in-flight requests per model instance.
	// Values above 10 are clamped to 10 due to Yuanqi API limits.
	MaxConcurrency int
	// MaxRetries controls retry attempts for retryable network/HTTP errors.
	// Use 0 to apply the default retry count.
	MaxRetries      int
	RetryBaseDelay  time.Duration
	CustomVariables map[string]string

	HTTPClient *http.Client
	Debugf     func(format string, args ...any)
}

// ChatModel is a Yuanqi provider implementing BaseChatModel and ToolCallingChatModel.
type ChatModel struct {
	baseURL      string
	appKey       string
	assistantID  string
	userID       string
	maxRetries   int
	retryDelay   time.Duration
	defaultVars  map[string]string
	client       *http.Client
	debugf       func(format string, args ...any)
	tools        []*schema.ToolInfo
	concurrencyC chan struct{}
}

var _ model.BaseChatModel = (*ChatModel)(nil)
var _ model.ToolCallingChatModel = (*ChatModel)(nil)

// NewChatModel creates a Yuanqi chat model.
func NewChatModel(_ context.Context, cfg *Config) (*ChatModel, error) {
	if cfg == nil {
		return nil, errors.New("nil config")
	}
	if strings.TrimSpace(cfg.AppKey) == "" {
		return nil, errors.New("app key is required")
	}
	if strings.TrimSpace(cfg.AssistantID) == "" {
		return nil, errors.New("assistant id is required")
	}
	if strings.TrimSpace(cfg.UserID) == "" {
		return nil, errors.New("user id is required")
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	maxConcurrency := cfg.MaxConcurrency
	if maxConcurrency <= 0 {
		maxConcurrency = defaultMaxConcurrency
	}
	if maxConcurrency > defaultMaxConcurrency {
		maxConcurrency = defaultMaxConcurrency
	}

	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = defaultTimeout
	}

	maxRetries := cfg.MaxRetries
	if maxRetries < 0 {
		return nil, errors.New("max retries cannot be negative")
	}
	if maxRetries == 0 {
		maxRetries = defaultMaxRetries
	}

	retryDelay := cfg.RetryBaseDelay
	if retryDelay <= 0 {
		retryDelay = defaultRetryDelay
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: timeout}
	} else if client.Timeout == 0 {
		client.Timeout = timeout
	}

	return &ChatModel{
		baseURL:      baseURL,
		appKey:       cfg.AppKey,
		assistantID:  cfg.AssistantID,
		userID:       cfg.UserID,
		maxRetries:   maxRetries,
		retryDelay:   retryDelay,
		defaultVars:  cloneMap(cfg.CustomVariables),
		client:       client,
		debugf:       cfg.Debugf,
		concurrencyC: make(chan struct{}, maxConcurrency),
	}, nil
}

// WithTools returns a new model instance with tools attached.
func (m *ChatModel) WithTools(tools []*schema.ToolInfo) (model.ToolCallingChatModel, error) {
	if m == nil {
		return nil, errors.New("nil model")
	}
	cloned := *m
	cloned.tools = append([]*schema.ToolInfo(nil), tools...)
	return &cloned, nil
}

// Generate sends a non-stream request to Yuanqi.
func (m *ChatModel) Generate(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.Message, error) {
	if err := m.acquire(ctx); err != nil {
		return nil, err
	}
	defer m.release()

	commonOpts := model.GetCommonOptions(nil, opts...)
	callOpts := model.GetImplSpecificOptions(&callOptions{}, opts...)
	m.logIgnoredCommonOptions(commonOpts)

	req, err := m.buildRequest(input, false, callOpts)
	if err != nil {
		return nil, err
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := m.doRequest(ctx, reqBody)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	msg, err := m.mapNonStreamResponse(&out)
	if err != nil {
		return nil, err
	}
	return msg, nil
}

// Stream sends a stream request to Yuanqi.
func (m *ChatModel) Stream(ctx context.Context, input []*schema.Message, opts ...model.Option) (*schema.StreamReader[*schema.Message], error) {
	if err := m.acquire(ctx); err != nil {
		return nil, err
	}

	commonOpts := model.GetCommonOptions(nil, opts...)
	callOpts := model.GetImplSpecificOptions(&callOptions{}, opts...)
	m.logIgnoredCommonOptions(commonOpts)

	req, err := m.buildRequest(input, true, callOpts)
	if err != nil {
		m.release()
		return nil, err
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		m.release()
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := m.doRequest(ctx, reqBody)
	if err != nil {
		m.release()
		return nil, err
	}

	sr, sw := schema.Pipe[*schema.Message](0)
	go func() {
		defer m.release()
		defer sw.Close()
		defer resp.Body.Close()

		if err := m.consumeStream(ctx, resp.Body, sw); err != nil {
			sw.Send(nil, err)
		}
	}()
	return sr, nil
}

func (m *ChatModel) consumeStream(ctx context.Context, body io.Reader, sw *schema.StreamWriter[*schema.Message]) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return err
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, ":") || strings.HasPrefix(line, "event:") {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
		if line == "" {
			continue
		}
		if line == "[DONE]" {
			return nil
		}
		if !strings.HasPrefix(line, "{") {
			continue
		}

		var out chatResponse
		if err := json.Unmarshal([]byte(line), &out); err != nil {
			return fmt.Errorf("decode stream chunk: %w", err)
		}

		msg, ok := m.mapStreamResponse(&out)
		if !ok {
			continue
		}
		if closed := sw.Send(msg, nil); closed {
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan stream: %w", err)
	}
	return nil
}

func (m *ChatModel) doRequest(ctx context.Context, body []byte) (*http.Response, error) {
	var lastErr error
	for attempt := 0; attempt <= m.maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL, bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("build request: %w", err)
		}
		req.Header.Set("Authorization", "Bearer "+m.appKey)
		req.Header.Set("Content-Type", "application/json")

		resp, err := m.client.Do(req)
		if err != nil {
			lastErr = err
			if !isRetryableErr(err) || attempt == m.maxRetries {
				return nil, fmt.Errorf("request failed: %w", err)
			}
			if waitErr := m.waitRetry(ctx, attempt); waitErr != nil {
				return nil, waitErr
			}
			continue
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return resp, nil
		}

		httpErr := readHTTPError(resp)
		lastErr = httpErr
		if !isRetryableStatus(resp.StatusCode) || attempt == m.maxRetries {
			return nil, httpErr
		}
		if waitErr := m.waitRetry(ctx, attempt); waitErr != nil {
			return nil, waitErr
		}
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("request failed")
}

func (m *ChatModel) waitRetry(ctx context.Context, attempt int) error {
	delay := m.retryDelay
	for i := 0; i < attempt; i++ {
		if delay >= maxRetryDelay/2 {
			delay = maxRetryDelay
			break
		}
		delay *= 2
	}
	if delay > maxRetryDelay {
		delay = maxRetryDelay
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (m *ChatModel) buildRequest(input []*schema.Message, stream bool, callOpts *callOptions) (*chatRequest, error) {
	if len(input) == 0 {
		return nil, errors.New("input messages are required")
	}

	msgs := make([]yuanqiMessage, 0, len(input))
	for i := range input {
		msg, err := mapRequestMessage(input[i])
		if err != nil {
			return nil, fmt.Errorf("map message %d: %w", i, err)
		}
		msgs = append(msgs, msg)
	}

	userID := m.userID
	if callOpts != nil && callOpts.UserID != nil {
		userID = *callOpts.UserID
	}
	if strings.TrimSpace(userID) == "" {
		return nil, errors.New("user id is empty")
	}

	customVars := cloneMap(m.defaultVars)
	if callOpts != nil {
		customVars = mergeMaps(customVars, callOpts.CustomVariables)
	}

	req := &chatRequest{
		AssistantID:     m.assistantID,
		UserID:          userID,
		Stream:          stream,
		Messages:        msgs,
		CustomVariables: customVars,
	}
	return req, nil
}

func mapRequestMessage(msg *schema.Message) (yuanqiMessage, error) {
	if msg == nil {
		return yuanqiMessage{}, errors.New("nil message")
	}

	role, err := toYuanqiRole(msg.Role)
	if err != nil {
		return yuanqiMessage{}, err
	}

	parts, err := toYuanqiContent(msg)
	if err != nil {
		return yuanqiMessage{}, err
	}
	if len(parts) == 0 {
		return yuanqiMessage{}, errors.New("empty message content")
	}

	return yuanqiMessage{
		Role:    role,
		Content: parts,
	}, nil
}

func toYuanqiRole(role schema.RoleType) (string, error) {
	switch role {
	case schema.User:
		return "user", nil
	case schema.Assistant:
		return "assistant", nil
	default:
		return "", fmt.Errorf("unsupported role: %s", role)
	}
}

func toYuanqiContent(msg *schema.Message) ([]yuanqiContent, error) {
	var out []yuanqiContent

	if len(msg.UserInputMultiContent) > 0 {
		for _, p := range msg.UserInputMultiContent {
			switch p.Type {
			case schema.ChatMessagePartTypeText:
				out = append(out, yuanqiContent{Type: "text", Text: p.Text})
			case schema.ChatMessagePartTypeImageURL:
				if p.Image == nil || p.Image.URL == nil {
					return nil, errors.New("image_url part missing url")
				}
				out = append(out, yuanqiContent{
					Type: "image_url",
					ImageURL: &yuanqiImageURL{
						Type: "image_url",
						URL:  *p.Image.URL,
					},
				})
			default:
				return nil, fmt.Errorf("unsupported multi content type: %s", p.Type)
			}
		}
		return out, nil
	}

	if len(msg.MultiContent) > 0 {
		for _, p := range msg.MultiContent {
			switch p.Type {
			case schema.ChatMessagePartTypeText:
				out = append(out, yuanqiContent{Type: "text", Text: p.Text})
			case schema.ChatMessagePartTypeImageURL:
				if p.ImageURL == nil || p.ImageURL.URL == "" {
					return nil, errors.New("image_url part missing url")
				}
				out = append(out, yuanqiContent{
					Type: "image_url",
					ImageURL: &yuanqiImageURL{
						Type: "image_url",
						URL:  p.ImageURL.URL,
					},
				})
			default:
				return nil, fmt.Errorf("unsupported deprecated multi content type: %s", p.Type)
			}
		}
		return out, nil
	}

	return []yuanqiContent{{
		Type: "text",
		Text: msg.Content,
	}}, nil
}

func (m *ChatModel) mapNonStreamResponse(resp *chatResponse) (*schema.Message, error) {
	if len(resp.Choices) == 0 {
		return nil, errors.New("empty choices")
	}
	c := resp.Choices[0]
	if c.Message == nil {
		return nil, errors.New("missing choices[0].message")
	}

	role, err := toSchemaRole(c.Message.Role)
	if err != nil {
		return nil, err
	}

	msg := &schema.Message{
		Role:      role,
		Content:   c.Message.Content,
		ToolCalls: extractToolCalls(c.Message.Steps, false),
		Extra: map[string]any{
			"assistant_id":     resp.AssistantID,
			"response_id":      resp.ID,
			"moderation_level": c.ModerationLevel,
		},
	}

	msg.ResponseMeta = &schema.ResponseMeta{
		FinishReason: c.FinishReason,
		Usage:        toTokenUsage(resp.Usage),
	}
	return msg, nil
}

func (m *ChatModel) mapStreamResponse(resp *chatResponse) (*schema.Message, bool) {
	if len(resp.Choices) == 0 {
		return nil, false
	}
	c := resp.Choices[0]
	if c.Delta == nil {
		return nil, false
	}

	msg := &schema.Message{
		Content:   c.Delta.Content,
		ToolCalls: extractToolCallsFromDelta(c.Delta.ToolCalls),
	}

	if c.Delta.Role != "" {
		role, err := toSchemaRole(c.Delta.Role)
		if err == nil {
			msg.Role = role
		}
	}

	if c.FinishReason != "" || resp.Usage != nil {
		msg.ResponseMeta = &schema.ResponseMeta{
			FinishReason: c.FinishReason,
			Usage:        toTokenUsage(resp.Usage),
		}
	}

	if resp.AssistantID != "" || resp.ID != "" || c.ModerationLevel != "" {
		msg.Extra = map[string]any{
			"assistant_id":     resp.AssistantID,
			"response_id":      resp.ID,
			"moderation_level": c.ModerationLevel,
		}
	}

	if msg.Role == "" && msg.Content == "" && len(msg.ToolCalls) == 0 && msg.ResponseMeta == nil && len(msg.Extra) == 0 {
		return nil, false
	}
	return msg, true
}

func extractToolCalls(steps []chatStep, withIndex bool) []schema.ToolCall {
	var out []schema.ToolCall
	for _, step := range steps {
		for i, tc := range step.ToolCalls {
			call := schema.ToolCall{
				ID:   tc.ID,
				Type: tc.Type,
				Function: schema.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
			if withIndex {
				call.Index = intPtr(i)
			}

			if tc.Function.Desc != "" || tc.Function.Type != "" {
				call.Extra = map[string]any{
					"function_desc": tc.Function.Desc,
					"function_type": tc.Function.Type,
				}
			}
			out = append(out, call)
		}
	}
	return out
}

func extractToolCallsFromDelta(calls []chatToolCall) []schema.ToolCall {
	out := make([]schema.ToolCall, 0, len(calls))
	for i, tc := range calls {
		call := schema.ToolCall{
			Index: intPtr(i),
			ID:    tc.ID,
			Type:  tc.Type,
			Function: schema.FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		}
		if tc.Function.Desc != "" || tc.Function.Type != "" {
			call.Extra = map[string]any{
				"function_desc": tc.Function.Desc,
				"function_type": tc.Function.Type,
			}
		}
		out = append(out, call)
	}
	return out
}

func toSchemaRole(role string) (schema.RoleType, error) {
	switch role {
	case "assistant":
		return schema.Assistant, nil
	case "user":
		return schema.User, nil
	default:
		return "", fmt.Errorf("unsupported role in response: %s", role)
	}
}

func toTokenUsage(u *chatUsage) *schema.TokenUsage {
	if u == nil {
		return nil
	}
	return &schema.TokenUsage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
}

func (m *ChatModel) acquire(ctx context.Context) error {
	select {
	case m.concurrencyC <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (m *ChatModel) release() {
	select {
	case <-m.concurrencyC:
	default:
	}
}

func (m *ChatModel) logIgnoredCommonOptions(opts *model.Options) {
	if m.debugf == nil || opts == nil {
		return
	}
	var ignored []string
	if opts.Temperature != nil {
		ignored = append(ignored, "temperature")
	}
	if opts.TopP != nil {
		ignored = append(ignored, "top_p")
	}
	if opts.MaxTokens != nil {
		ignored = append(ignored, "max_tokens")
	}
	if opts.Model != nil {
		ignored = append(ignored, "model")
	}
	if len(opts.Stop) > 0 {
		ignored = append(ignored, "stop")
	}
	if len(opts.Tools) > 0 {
		ignored = append(ignored, "tools")
	}
	if opts.ToolChoice != nil {
		ignored = append(ignored, "tool_choice")
	}
	if len(opts.AllowedToolNames) > 0 {
		ignored = append(ignored, "allowed_tool_names")
	}
	if len(ignored) > 0 {
		m.debugf("yuanqi provider ignored unsupported options: %s", strings.Join(ignored, ","))
	}
}

type HTTPError struct {
	StatusCode int
	RequestID  string
	Body       string
}

func (e *HTTPError) Error() string {
	if e.RequestID != "" {
		return fmt.Sprintf("yuanqi request failed: status=%d request_id=%s body=%s", e.StatusCode, e.RequestID, e.Body)
	}
	return fmt.Sprintf("yuanqi request failed: status=%d body=%s", e.StatusCode, e.Body)
}

func readHTTPError(resp *http.Response) error {
	defer resp.Body.Close()

	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodyBytes))
	requestID := resp.Header.Get("X-Request-Id")
	if requestID == "" {
		requestID = resp.Header.Get("x-request-id")
	}
	return &HTTPError{
		StatusCode: resp.StatusCode,
		RequestID:  requestID,
		Body:       strings.TrimSpace(string(body)),
	}
}

func isRetryableStatus(statusCode int) bool {
	return statusCode == http.StatusTooManyRequests || statusCode >= 500
}

func isRetryableErr(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	return true
}

func cloneMap(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	dst := make(map[string]string, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func mergeMaps(base, override map[string]string) map[string]string {
	if len(base) == 0 && len(override) == 0 {
		return nil
	}
	merged := cloneMap(base)
	if merged == nil {
		merged = make(map[string]string, len(override))
	}
	for k, v := range override {
		merged[k] = v
	}
	return merged
}

func intPtr(v int) *int {
	return &v
}

type chatRequest struct {
	AssistantID     string            `json:"assistant_id"`
	UserID          string            `json:"user_id"`
	Stream          bool              `json:"stream"`
	Messages        []yuanqiMessage   `json:"messages"`
	CustomVariables map[string]string `json:"custom_variables,omitempty"`
}

type yuanqiMessage struct {
	Role    string          `json:"role"`
	Content []yuanqiContent `json:"content"`
}

type yuanqiContent struct {
	Type     string          `json:"type,omitempty"`
	Text     string          `json:"text,omitempty"`
	ImageURL *yuanqiImageURL `json:"image_url,omitempty"`
}

type yuanqiImageURL struct {
	Type string `json:"type,omitempty"`
	URL  string `json:"url,omitempty"`
}

type chatResponse struct {
	ID          string       `json:"id"`
	AssistantID string       `json:"assistant_id"`
	Choices     []chatChoice `json:"choices"`
	Usage       *chatUsage   `json:"usage"`
}

type chatChoice struct {
	Index           int          `json:"index"`
	FinishReason    string       `json:"finish_reason"`
	ModerationLevel string       `json:"moderation_level"`
	Message         *chatMessage `json:"message"`
	Delta           *chatDelta   `json:"delta"`
}

type chatMessage struct {
	Role    string     `json:"role"`
	Content string     `json:"content"`
	Steps   []chatStep `json:"steps"`
}

type chatStep struct {
	Role       string         `json:"role"`
	Content    string         `json:"content"`
	ToolCallID string         `json:"tool_call_id"`
	ToolCalls  []chatToolCall `json:"tool_calls"`
}

type chatDelta struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	ToolCalls []chatToolCall `json:"tool_calls"`
}

type chatToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function chatFunction `json:"function"`
}

type chatFunction struct {
	Name      string `json:"name"`
	Desc      string `json:"desc"`
	Type      string `json:"type"`
	Arguments string `json:"arguments"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
