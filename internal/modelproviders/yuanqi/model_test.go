package yuanqi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/cloudwego/eino/schema"
)

func TestGenerateMappingAndOptions(t *testing.T) {
	t.Parallel()

	var gotReq chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "Bearer test-app-key", r.Header.Get("Authorization"))
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotReq))

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
  "id":"resp-1",
  "assistant_id":"asst-1",
  "choices":[
    {
      "index":0,
      "finish_reason":"stop",
      "moderation_level":"1",
      "message":{
        "role":"assistant",
        "content":"pong",
        "steps":[
          {
            "role":"assistant",
            "tool_calls":[
              {
                "id":"call_1",
                "type":"function",
                "function":{
                  "name":"lookup",
                  "arguments":"{\"k\":\"v\"}"
                }
              }
            ]
          }
        ]
      }
    }
  ],
  "usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
}`))
	}))
	defer server.Close()

	m, err := NewChatModel(context.Background(), &Config{
		BaseURL:         server.URL,
		AppKey:          "test-app-key",
		AssistantID:     "asst-1",
		UserID:          "default-user",
		CustomVariables: map[string]string{"A": "1", "B": "0"},
		MaxRetries:      1,
	})
	require.NoError(t, err)

	msg, err := m.Generate(context.Background(), []*schema.Message{
		{Role: schema.User, Content: "ping"},
	}, WithUserID("u-override"), WithCustomVariables(map[string]string{"B": "2"}))
	require.NoError(t, err)

	require.Equal(t, "asst-1", gotReq.AssistantID)
	require.Equal(t, "u-override", gotReq.UserID)
	require.Equal(t, map[string]string{"A": "1", "B": "2"}, gotReq.CustomVariables)
	require.False(t, gotReq.Stream)
	require.Len(t, gotReq.Messages, 1)
	require.Equal(t, "user", gotReq.Messages[0].Role)
	require.Equal(t, "ping", gotReq.Messages[0].Content[0].Text)

	require.Equal(t, schema.Assistant, msg.Role)
	require.Equal(t, "pong", msg.Content)
	require.NotNil(t, msg.ResponseMeta)
	require.Equal(t, "stop", msg.ResponseMeta.FinishReason)
	require.NotNil(t, msg.ResponseMeta.Usage)
	require.Equal(t, 7, msg.ResponseMeta.Usage.TotalTokens)
	require.Equal(t, "1", msg.Extra["moderation_level"])
	require.Len(t, msg.ToolCalls, 1)
	require.Equal(t, "lookup", msg.ToolCalls[0].Function.Name)
}

func TestStreamConcatable(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"id":"resp-2","assistant_id":"asst-2","choices":[{"index":0,"delta":{"role":"assistant","content":"hel"}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"id":"resp-2","assistant_id":"asst-2","choices":[{"index":0,"delta":{"content":"lo","tool_calls":[{"id":"tool_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\""}}]}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"id":"resp-2","assistant_id":"asst-2","choices":[{"index":0,"finish_reason":"stop","delta":{"tool_calls":[{"id":"tool_1","type":"function","function":{"name":"weather","arguments":"beijing\"}"}}]}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}` + "\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer server.Close()

	m, err := NewChatModel(context.Background(), &Config{
		BaseURL:     server.URL,
		AppKey:      "test-app-key",
		AssistantID: "asst-2",
		UserID:      "user-1",
		MaxRetries:  1,
	})
	require.NoError(t, err)

	sr, err := m.Stream(context.Background(), []*schema.Message{
		{Role: schema.User, Content: "hello"},
	})
	require.NoError(t, err)
	defer sr.Close()

	var chunks []*schema.Message
	for {
		chunk, recvErr := sr.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}
		require.NoError(t, recvErr)
		chunks = append(chunks, chunk)
	}

	merged, err := schema.ConcatMessages(chunks)
	require.NoError(t, err)
	require.Equal(t, schema.Assistant, merged.Role)
	require.Equal(t, "hello", merged.Content)
	require.NotNil(t, merged.ResponseMeta)
	require.Equal(t, "stop", merged.ResponseMeta.FinishReason)
	require.Equal(t, 3, merged.ResponseMeta.Usage.TotalTokens)
	require.Len(t, merged.ToolCalls, 1)
	require.Equal(t, "weather", merged.ToolCalls[0].Function.Name)
	require.Equal(t, `{"city":"beijing"}`, merged.ToolCalls[0].Function.Arguments)
}

func TestGenerateRetry(t *testing.T) {
	t.Parallel()

	var calls int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt32(&calls, 1)
		if n == 1 {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = w.Write([]byte("bad gateway"))
			return
		}
		_, _ = w.Write([]byte(`{"id":"ok","choices":[{"index":0,"message":{"role":"assistant","content":"ok"}}]}`))
	}))
	defer server.Close()

	m, err := NewChatModel(context.Background(), &Config{
		BaseURL:        server.URL,
		AppKey:         "test-app-key",
		AssistantID:    "asst",
		UserID:         "user",
		MaxRetries:     1,
		RetryBaseDelay: 1 * time.Millisecond,
	})
	require.NoError(t, err)

	msg, err := m.Generate(context.Background(), []*schema.Message{{Role: schema.User, Content: "x"}})
	require.NoError(t, err)
	require.Equal(t, "ok", msg.Content)
	require.Equal(t, int32(2), atomic.LoadInt32(&calls))
}

func TestWithToolsReturnsNewInstance(t *testing.T) {
	t.Parallel()

	m, err := NewChatModel(context.Background(), &Config{
		AppKey:      "test-app-key",
		AssistantID: "asst",
		UserID:      "user",
	})
	require.NoError(t, err)

	cloneAny, err := m.WithTools([]*schema.ToolInfo{{Name: "t1"}})
	require.NoError(t, err)

	clone, ok := cloneAny.(*ChatModel)
	require.True(t, ok)
	require.NotSame(t, m, clone)
	require.Len(t, clone.tools, 1)
	require.Len(t, m.tools, 0)
}
