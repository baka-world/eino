package yuanqi

import "github.com/cloudwego/eino/components/model"

type callOptions struct {
	UserID          *string
	CustomVariables map[string]string
}

// WithUserID overrides the configured default user_id for one model call.
func WithUserID(userID string) model.Option {
	return model.WrapImplSpecificOptFn(func(o *callOptions) {
		o.UserID = &userID
	})
}

// WithCustomVariables sets custom_variables for one model call.
func WithCustomVariables(customVariables map[string]string) model.Option {
	return model.WrapImplSpecificOptFn(func(o *callOptions) {
		o.CustomVariables = cloneMap(customVariables)
	})
}
