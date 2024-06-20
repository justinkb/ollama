package template

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"text/template"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestFuncs(t *testing.T) {
	t.Run("toJson", func(t *testing.T) {
		cases := []struct {
			input    any
			expected string
		}{
			{nil, "null"},
			{true, "true"},
			{false, "false"},
			{0, "0"},
			{1, "1"},
			{1.0, "1"},
			{1.1, "1.1"},
			{"", `""`},
			{"hello", `"hello"`},
			{[]int{1, 2, 3}, "[1,2,3]"},
			{[]string{"a", "b", "c"}, `["a","b","c"]`},
			{map[string]int{"a": 1, "b": 2}, `{"a":1,"b":2}`},
			{map[string]string{"a": "b", "c": "d"}, `{"a":"b","c":"d"}`},
		}

		for _, tt := range cases {
			t.Run(tt.expected, func(t *testing.T) {
				toJson, ok := funcs["toJson"].(func(any) string)
				if !ok {
					t.Fatal("toJson is not a function")
				}

				if s := toJson(tt.input); s != tt.expected {
					t.Errorf("expected %q, got %q", tt.expected, s)
				}
			})
		}
	})
}

func TestNamed(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "templates.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var ss map[string]string
		if err := json.Unmarshal(scanner.Bytes(), &ss); err != nil {
			t.Fatal(err)
		}

		for k, v := range ss {
			t.Run(k, func(t *testing.T) {
				kv := llm.KV{"tokenizer.chat_template": v}
				s := kv.ChatTemplate()
				r, err := Named(s)
				if err != nil {
					t.Fatal(err)
				}

				if r.Name != k {
					t.Errorf("expected %q, got %q", k, r.Name)
				}

				var b bytes.Buffer
				if _, err := io.Copy(&b, r.Reader()); err != nil {
					t.Fatal(err)
				}

				tmpl, err := template.New(s).Parse(b.String())
				if err != nil {
					t.Fatal(err)
				}

				if tmpl.Tree.Root.String() == "" {
					t.Errorf("empty %s template", k)
				}
			})
		}
	}
}

func TestParse(t *testing.T) {
	cases := []struct {
		template string
		vars     []string
	}{
		{"{{ .Prompt }}", []string{"prompt", "response"}},
		{"{{ .System }} {{ .Prompt }}", []string{"prompt", "response", "system"}},
		{"{{ .System }} {{ .Prompt }} {{ .Response }}", []string{"prompt", "response", "system"}},
		{"{{ with .Tools }}{{ . }}{{ end }} {{ .System }} {{ .Prompt }}", []string{"prompt", "response", "system", "tools"}},
		{"{{ range .Messages }}{{ .Role }} {{ .Content }}{{ end }}", []string{"content", "messages", "role"}},
		{"{{ range .Messages }}{{ if eq .Role \"system\" }}SYSTEM: {{ .Content }}{{ else if eq .Role \"user\" }}USER: {{ .Content }}{{ else if eq .Role \"assistant\" }}ASSISTANT: {{ .Content }}{{ end }}{{ end }}", []string{"content", "messages", "role"}},
	}

	for _, tt := range cases {
		t.Run("", func(t *testing.T) {
			tmpl, err := Parse(tt.template)
			if err != nil {
				t.Fatal(err)
			}

			vars := tmpl.Vars()
			if !slices.Equal(tt.vars, vars) {
				t.Errorf("expected %v, got %v", tt.vars, vars)
			}
		})
	}
}

func TestExecuteWithMessages(t *testing.T) {
	type template struct {
		name     string
		template string
	}
	cases := []struct {
		name      string
		templates []template
		values    Values
		expected  string
	}{
		{
			"mistral",
			[]template{
				{"no response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] `},
				{"response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`},
				{"messages", `{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ if and ($.Messages.Last "user" .) $.System }}{{ $.System }}{{ "\n\n" }}
{{- end }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}
{{- end }}
{{- end }}`},
			},
			Values{
				Messages: []api.Message{
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "Yay!"},
				},
			},
			`[INST] Hello friend![/INST] Hello human![INST] Yay![/INST] `,
		},
		{
			"mistral system",
			[]template{
				{"no response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] `},
				{"response", `[INST] {{ if .System }}{{ .System }}{{ "\n\n" }}{{ end }}{{ .Prompt }}[/INST] {{ .Response }}`},
				{"messages", `
{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ if and ($.Messages.Last "user" .) $.System }}{{ $.System }}{{ "\n\n" }}
{{- end }}{{ .Content }}[/INST] {{ else if eq .Role "assistant" }}{{ .Content }}
{{- end }}
{{- end }}`},
			},
			Values{
				Messages: []api.Message{
					{Role: "system", Content: "You are a helpful assistant!"},
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "Yay!"},
				},
			},
			`[INST] Hello friend![/INST] Hello human![INST] You are a helpful assistant!

Yay![/INST] `,
		},
		{
			"chatml",
			[]template{
				// this does not have a "no response" test because it's impossible to render the same output
				{"response", `{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
`},
				{"messages", `
{{- range .Messages }}
{{- if and (eq .Role "user") ($.Messages.Last "user" .) $.System }}<|im_start|>system
{{ $.System }}<|im_end|>{{ "\n" }}
{{- end }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>{{ "\n" }}
{{- end }}<|im_start|>assistant
`},
			},
			Values{
				Messages: []api.Message{
					{Role: "system", Content: "You are a helpful assistant!"},
					{Role: "user", Content: "Hello friend!"},
					{Role: "assistant", Content: "Hello human!"},
					{Role: "user", Content: "Yay!"},
				},
			},
			`<|im_start|>user
Hello friend!<|im_end|>
<|im_start|>assistant
Hello human!<|im_end|>
<|im_start|>system
You are a helpful assistant!<|im_end|>
<|im_start|>user
Yay!<|im_end|>
<|im_start|>assistant
`,
		},
		{
			"moondream",
			[]template{
				// this does not have a "no response" test because it's impossible to render the same output
				{"response", `{{ if .Prompt }}Question: {{ .Prompt }}

{{ end }}Answer: {{ .Response }}

`},
				{"messages", `
{{- range .Messages }}
{{- if eq .Role "user" }}Question: {{ .Content }}{{ "\n\n" }}
{{- else if eq .Role "assistant" }}Answer: {{ .Content }}{{ "\n\n" }}
{{- end }}
{{- end }}Answer: `},
			},
			Values{
				Messages: []api.Message{
					{Role: "user", Content: "What's in this image?", Images: []api.ImageData{[]byte("")}},
					{Role: "assistant", Content: "It's a hot dog."},
					{Role: "user", Content: "What's in _this_ image?"},
					{Role: "user", Images: []api.ImageData{[]byte("")}},
					{Role: "user", Content: "Is it a hot dog?"},
				},
			},
			`Question: [img-0] What's in this image?

Answer: It's a hot dog.

Question: What's in _this_ image?

[img-1]

Is it a hot dog?

Answer: `,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			for _, ttt := range tt.templates {
				t.Run(ttt.name, func(t *testing.T) {
					tmpl, err := Parse(ttt.template)
					if err != nil {
						t.Fatal(err)
					}

					var b bytes.Buffer
					if err := tmpl.Execute(&b, tt.values); err != nil {
						t.Fatal(err)
					}

					if b.String() != tt.expected {
						t.Errorf("expected\n%s,\ngot\n%s", tt.expected, b.String())
					}
				})
			}
		})
	}
}

func TestMessagesLast(t *testing.T) {
	s := messages{
		{Role: "user", Content: "What have I got in my pocket?"},
		{Role: "assistant", Content: "Not fair! not fair! It isn't fair, my precious, is it, to ask us what it's got in its nassty little pocketses?"},
		{Role: "user", Content: "What have I got in my pocket?"},
		{Role: "assistant", Content: "It must give us three guesseses, my precious, three guesseses."},
		{Role: "user", Content: "Very well! Guess away!"},
		{Role: "assistant", Content: "Handses!"},
		{Role: "user", Content: "Guess again!"},
	}

	cases := []struct {
		role   string
		expect bool
	}{
		{"user", false},
		{"assistant", false},
		{"user", false},
		{"assistant", false},
		{"user", false},
		{"assistant", true},
		{"user", true},
	}

	for i, tt := range cases {
		t.Run(fmt.Sprintf("%s-%d", tt.role, i), func(t *testing.T) {
			if actual := s.Last(tt.role, s[i]); actual != tt.expect {
				t.Fatalf("expected %v got %v", tt.expect, actual)
			}
		})
	}
}
