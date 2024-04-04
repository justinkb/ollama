package model

import (
	"fmt"
	"strings"
	"testing"
)

var testPaths = map[string]Path{
	"mistral:latest":      {name: "mistral", tag: "latest"},
	"mistral":             {name: "mistral"},
	"mistral:30B":         {name: "mistral", tag: "30B"},
	"mistral:7b":          {name: "mistral", tag: "7b"},
	"mistral:7b+Q4_0":     {name: "mistral", tag: "7b", build: "Q4_0"},
	"mistral+KQED":        {name: "mistral", build: "KQED"},
	"mistral.x-3:7b+Q4_0": {name: "mistral.x-3", tag: "7b", build: "Q4_0"},
	"mistral:7b+q4_0":     {name: "mistral", tag: "7b", build: "Q4_0"},
	"llama2":              {name: "llama2"},

	// invalid (includes fuzzing trophies)
	"+":                      {},
	"mistral:7b+Q4_0:latest": {},
	"mi tral":                {},
	"x/y/z/foo":              {},
	"/0":                     {},
	"0 /0":                   {},
	"0 /":                    {},
	"0/":                     {},
	":":                      {},
	":/0":                    {},
	"+0/00000":               {},
	"0+.\xf2\x80\xf6\x9d00000\xe5\x99\xe6\xd900\xd90\xa60\x91\xdc0\xff\xbf\x99\xe800\xb9\xdc\xd6\xc300\x970\xfb\xfd0\xe0\x8a\xe1\xad\xd40\x9700\xa80\x980\xdd0000\xb00\x91000\xfe0\x89\x9b\x90\x93\x9f0\xe60\xf7\x84\xb0\x87\xa5\xff0\xa000\x9a\x85\xf6\x85\xfe\xa9\xf9\xe9\xde00\xf4\xe0\x8f\x81\xad\xde00\xd700\xaa\xe000000\xb1\xee0\x91": {},
	"0//0":                        {},
	"m+^^^":                       {},
	"file:///etc/passwd":          {},
	"file:///etc/passwd:latest":   {},
	"file:///etc/passwd:latest+u": {},

	strings.Repeat("a", MaxPathLength):   {name: strings.Repeat("a", MaxPathLength)},
	strings.Repeat("a", MaxPathLength+1): {},
}

func TestPathParts(t *testing.T) {
	const wantNumParts = 5
	var p Path
	if len(p.Parts()) != wantNumParts {
		t.Errorf("Parts() = %d; want %d", len(p.Parts()), wantNumParts)
	}
}

func TestParsePath(t *testing.T) {
	for s, want := range testPaths {
		for _, prefix := range []string{"", "https://", "http://"} {
			// We should get the same results with or without the
			// http(s) prefixes
			s := prefix + s

			t.Run(s, func(t *testing.T) {
				got := ParsePath(s)
				if got != want {
					t.Errorf("ParsePath(%q) = %q; want %q", s, got, want)
				}

				// test round-trip
				if ParsePath(got.String()) != got {
					t.Errorf("String() = %s; want %s", got.String(), s)
				}

				if got.Valid() && got.Name() == "" {
					t.Errorf("Valid() = true; Name() = %q; want non-empty name", got.Name())
				} else if !got.Valid() && got.Name() != "" {
					t.Errorf("Valid() = false; Name() = %q; want empty name", got.Name())
				}
			})
		}
	}
}

func TestPathComplete(t *testing.T) {
	cases := []struct {
		in                   string
		complete             bool
		completeWithoutBuild bool
	}{
		{"", false, false},
		{"example.com/mistral:7b+x", false, false},
		{"example.com/mistral:7b+Q4_0", false, false},
		{"mistral:7b+x", false, false},
		{"example.com/x/mistral:latest+Q4_0", true, true},
		{"example.com/x/mistral:latest", false, true},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			p := ParsePath(tt.in)
			t.Logf("ParsePath(%q) = %#v", tt.in, p)
			if g := p.Complete(); g != tt.complete {
				t.Errorf("Complete(%q) = %v; want %v", tt.in, g, tt.complete)
			}
			if g := p.CompleteWithoutBuild(); g != tt.completeWithoutBuild {
				t.Errorf("CompleteWithoutBuild(%q) = %v; want %v", tt.in, g, tt.completeWithoutBuild)
			}
		})
	}
}

func TestPathStringVariants(t *testing.T) {
	cases := []struct {
		in              string
		nameAndTag      string
		nameTagAndBuild string
	}{
		{"x/y/z:8n+I", "z:8n", "z:8n+I"},
		{"x/y/z:8n", "z:8n", "z:8n"},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			p := ParsePath(tt.in)
			t.Logf("ParsePath(%q) = %#v", tt.in, p)
			if g := p.NameAndTag(); g != tt.nameAndTag {
				t.Errorf("NameAndTag(%q) = %q; want %q", tt.in, g, tt.nameAndTag)
			}
			if g := p.NameTagAndBuild(); g != tt.nameTagAndBuild {
				t.Errorf("NameTagAndBuild(%q) = %q; want %q", tt.in, g, tt.nameTagAndBuild)
			}
		})
	}
}

func TestPathFull(t *testing.T) {
	const empty = "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/!(MISSING NAME):!(MISSING TAG)+!(MISSING BUILD)"

	cases := []struct {
		in       string
		wantFull string
	}{
		{"", empty},
		{"example.com/mistral:7b+x", "!(MISSING DOMAIN)/example.com/mistral:7b+X"},
		{"example.com/mistral:7b+Q4_0", "!(MISSING DOMAIN)/example.com/mistral:7b+Q4_0"},
		{"example.com/x/mistral:latest", "example.com/x/mistral:latest+!(MISSING BUILD)"},
		{"example.com/x/mistral:latest+Q4_0", "example.com/x/mistral:latest+Q4_0"},

		{"mistral:7b+x", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+X"},
		{"mistral:7b+q4_0", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+Q4_0"},
		{"mistral:7b+Q4_0", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:7b+Q4_0"},
		{"mistral:latest", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:latest+!(MISSING BUILD)"},
		{"mistral", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:!(MISSING TAG)+!(MISSING BUILD)"},
		{"mistral:30b", "!(MISSING DOMAIN)/!(MISSING NAMESPACE)/mistral:30b+!(MISSING BUILD)"},
	}

	for _, tt := range cases {
		t.Run(tt.in, func(t *testing.T) {
			p := ParsePath(tt.in)
			t.Logf("ParsePath(%q) = %#v", tt.in, p)
			if g := p.Full(); g != tt.wantFull {
				t.Errorf("Full(%q) = %q; want %q", tt.in, g, tt.wantFull)
			}
		})
	}
}

func TestParsePathAllocs(t *testing.T) {
	// test allocations
	var r Path
	allocs := testing.AllocsPerRun(1000, func() {
		r = ParsePath("example.com/mistral:7b+Q4_0")
	})
	_ = r
	if allocs > 0 {
		t.Errorf("ParsePath allocs = %v; want 0", allocs)
	}
}

func BenchmarkParsePath(b *testing.B) {
	b.ReportAllocs()

	var r Path
	for i := 0; i < b.N; i++ {
		r = ParsePath("example.com/mistral:7b+Q4_0")
	}
	_ = r
}

func FuzzParsePath(f *testing.F) {
	f.Add("example.com/mistral:7b+Q4_0")
	f.Add("example.com/mistral:7b+q4_0")
	f.Add("example.com/mistral:7b+x")
	f.Add("x/y/z:8n+I")
	f.Fuzz(func(t *testing.T, s string) {
		r0 := ParsePath(s)
		if !r0.Valid() {
			if r0 != (Path{}) {
				t.Errorf("expected invalid path to be zero value; got %#v", r0)
			}
			t.Skipf("invalid path: %q", s)
		}

		for _, p := range r0.Parts() {
			if len(p) > MaxPathLength {
				t.Errorf("part too long: %q", p)
			}
		}

		if !strings.EqualFold(r0.String(), s) {
			t.Errorf("String() did not round-trip with case insensitivity: %q\ngot  = %q\nwant = %q", s, r0.String(), s)
		}

		r1 := ParsePath(r0.String())
		if r0 != r1 {
			t.Errorf("round-trip mismatch: %+v != %+v", r0, r1)
		}

	})
}

func ExampleMerge() {
	r := Merge(
		ParsePath("mistral"),
		ParsePath("registry.ollama.com/XXXXX:latest+Q4_0"),
	)
	fmt.Println(r)

	// Output:
	// registry.ollama.com/mistral:latest+Q4_0
}