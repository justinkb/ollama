// openai package provides middleware for partial compatibility with the OpenAI REST API
package openai

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

type Error struct {
	Message string      `json:"message"`
	Type    string      `json:"type"`
	Param   interface{} `json:"param"`
	Code    *string     `json:"code"`
}

type ErrorResponse struct {
	Error Error `json:"error"`
}

type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason *string `json:"finish_reason"`
}

type ChunkChoice struct {
	Index        int     `json:"index"`
	Delta        Message `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ResponseFormat struct {
	Type string `json:"type"`
}

type ChatCompletionRequest struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Stream           bool            `json:"stream"`
	MaxTokens        *int            `json:"max_tokens"`
	Seed             *int            `json:"seed"`
	Stop             any             `json:"stop"`
	Temperature      *float64        `json:"temperature"`
	FrequencyPenalty *float64        `json:"frequency_penalty"`
	PresencePenalty  *float64        `json:"presence_penalty_penalty"`
	TopP             *float64        `json:"top_p"`
	ResponseFormat   *ResponseFormat `json:"response_format"`
}

type ChatCompletion struct {
	Id                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage,omitempty"`
}

type ChatCompletionChunk struct {
	Id                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	SystemFingerprint string        `json:"system_fingerprint"`
	Choices           []ChunkChoice `json:"choices"`
}

type Model struct {
	Id      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ListCompletion struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

func NewError(code int, message string) ErrorResponse {
	var etype string
	switch code {
	case http.StatusBadRequest:
		etype = "invalid_request_error"
	case http.StatusNotFound:
		etype = "not_found_error"
	default:
		etype = "api_error"
	}

	return ErrorResponse{Error{Type: etype, Message: message}}
}

func toChatCompletion(id string, r api.ChatResponse) ChatCompletion {
	return ChatCompletion{
		Id:                id,
		Object:            "chat.completion",
		Created:           r.CreatedAt.Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []Choice{{
			Index:   0,
			Message: Message{Role: r.Message.Role, Content: r.Message.Content},
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
		}},
		Usage: Usage{
			// TODO: ollama returns 0 for prompt eval if the prompt was cached, but openai returns the actual count
			PromptTokens:     r.PromptEvalCount,
			CompletionTokens: r.EvalCount,
			TotalTokens:      r.PromptEvalCount + r.EvalCount,
		},
	}
}

func toChunk(id string, r api.ChatResponse) ChatCompletionChunk {
	return ChatCompletionChunk{
		Id:                id,
		Object:            "chat.completion.chunk",
		Created:           time.Now().Unix(),
		Model:             r.Model,
		SystemFingerprint: "fp_ollama",
		Choices: []ChunkChoice{{
			Index: 0,
			Delta: Message{Role: "assistant", Content: r.Message.Content},
			FinishReason: func(reason string) *string {
				if len(reason) > 0 {
					return &reason
				}
				return nil
			}(r.DoneReason),
		}},
	}
}

func toListCompletion(r api.ListResponse) ListCompletion {
	var data []Model
	for _, m := range r.Models {
		data = append(data, Model{
			Id:      m.Name,
			Object:  "model",
			Created: m.ModifiedAt.Unix(),
			OwnedBy: model.ParseName(m.Name).Namespace,
		})
	}

	return ListCompletion{
		Object: "list",
		Data:   data,
	}
}

func fromRequest(r ChatCompletionRequest) (api.ChatRequest, error) {
	var messages []api.Message
	for _, msg := range r.Messages {
		switch content := msg.Content.(type) {
		case string:
			messages = append(messages, api.Message{Role: msg.Role, Content: content})
		case []any:
			message := api.Message{Role: msg.Role}
			for _, c := range content {
				if data, ok := c.(map[string]any); ok {
					switch data["type"] {
					case "text":
						if text, ok := data["text"].(string); ok {
							message.Content = text
						} else {
							return api.ChatRequest{}, fmt.Errorf("invalid message format")
						}
					case "image_url":
						if urlMap, ok := data["image_url"].(map[string]any); ok {
							if url, ok := urlMap["url"].(string); ok {
								types := []string{"jpeg", "jpg", "png"}
								for _, t := range types {
									url = strings.TrimPrefix(url, "data:image/"+t+";base64,")
								}
								if img, err := base64.StdEncoding.DecodeString(url); err == nil {
									message.Images = append(message.Images, img)
								}
							} else {
								return api.ChatRequest{}, fmt.Errorf("invalid message format")
							}
						} else {
							return api.ChatRequest{}, fmt.Errorf("invalid message format")
						}
					default:
						return api.ChatRequest{}, fmt.Errorf("invalid message format")
					}
				} else {
					return api.ChatRequest{}, fmt.Errorf("invalid message format")
				}
			}
			if message.Content != "" || len(message.Images) > 0 {
				messages = append(messages, message)
			}
		default:
			return api.ChatRequest{}, fmt.Errorf("invalid message content type: %T", content)
		}
	}

	options := make(map[string]interface{})

	switch stop := r.Stop.(type) {
	case string:
		options["stop"] = []string{stop}
	case []interface{}:
		var stops []string
		for _, s := range stop {
			if str, ok := s.(string); ok {
				stops = append(stops, str)
			}
		}
		options["stop"] = stops
	}

	if r.MaxTokens != nil {
		options["num_predict"] = *r.MaxTokens
	}

	if r.Temperature != nil {
		options["temperature"] = *r.Temperature * 2.0
	} else {
		options["temperature"] = 1.0
	}

	if r.Seed != nil {
		options["seed"] = *r.Seed
	}

	if r.FrequencyPenalty != nil {
		options["frequency_penalty"] = *r.FrequencyPenalty * 2.0
	}

	if r.PresencePenalty != nil {
		options["presence_penalty"] = *r.PresencePenalty * 2.0
	}

	if r.TopP != nil {
		options["top_p"] = *r.TopP
	} else {
		options["top_p"] = 1.0
	}

	var format string
	if r.ResponseFormat != nil && r.ResponseFormat.Type == "json_object" {
		format = "json"
	}

	return api.ChatRequest{
		Model:    r.Model,
		Messages: messages,
		Format:   format,
		Options:  options,
		Stream:   &r.Stream,
	}, nil
}

type BaseWriter struct {
	gin.ResponseWriter
}

type ChatWriter struct {
	stream bool
	id     string
	BaseWriter
}

type ListWriter struct {
	BaseWriter
}

func (w *BaseWriter) writeError(code int, data []byte) (int, error) {
	var serr api.StatusError
	err := json.Unmarshal(data, &serr)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(NewError(http.StatusInternalServerError, serr.Error()))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) writeResponse(data []byte) (int, error) {
	var chatResponse api.ChatResponse
	err := json.Unmarshal(data, &chatResponse)
	if err != nil {
		return 0, err
	}

	// chat chunk
	if w.stream {
		d, err := json.Marshal(toChunk(w.id, chatResponse))
		if err != nil {
			return 0, err
		}

		w.ResponseWriter.Header().Set("Content-Type", "text/event-stream")
		_, err = w.ResponseWriter.Write([]byte(fmt.Sprintf("data: %s\n\n", d)))
		if err != nil {
			return 0, err
		}

		if chatResponse.Done {
			_, err = w.ResponseWriter.Write([]byte("data: [DONE]\n\n"))
			if err != nil {
				return 0, err
			}
		}

		return len(data), nil
	}

	// chat completion
	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(toChatCompletion(w.id, chatResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ChatWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(code, data)
	}

	return w.writeResponse(data)
}

func (w *ListWriter) writeResponse(data []byte) (int, error) {
	var listResponse api.ListResponse
	err := json.Unmarshal(data, &listResponse)
	if err != nil {
		return 0, err
	}

	w.ResponseWriter.Header().Set("Content-Type", "application/json")
	err = json.NewEncoder(w.ResponseWriter).Encode(toListCompletion(listResponse))
	if err != nil {
		return 0, err
	}

	return len(data), nil
}

func (w *ListWriter) Write(data []byte) (int, error) {
	code := w.ResponseWriter.Status()
	if code != http.StatusOK {
		return w.writeError(code, data)
	}

	return w.writeResponse(data)
}

func ListMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		w := &ListWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
		}

		c.Writer = w

		c.Next()
	}
}

func Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ChatCompletionRequest
		err := c.ShouldBindJSON(&req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
			return
		}

		if len(req.Messages) == 0 {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, "[] is too short - 'messages'"))
			return
		}

		var b bytes.Buffer

		chatReq, err := fromRequest(req)
		if err != nil {
			c.AbortWithStatusJSON(http.StatusBadRequest, NewError(http.StatusBadRequest, err.Error()))
		}

		if err := json.NewEncoder(&b).Encode(chatReq); err != nil {
			c.AbortWithStatusJSON(http.StatusInternalServerError, NewError(http.StatusInternalServerError, err.Error()))
			return
		}

		c.Request.Body = io.NopCloser(&b)

		w := &ChatWriter{
			BaseWriter: BaseWriter{ResponseWriter: c.Writer},
			stream:     req.Stream,
			id:         fmt.Sprintf("chatcmpl-%d", rand.Intn(999)),
		}

		c.Writer = w

		c.Next()
	}
}
