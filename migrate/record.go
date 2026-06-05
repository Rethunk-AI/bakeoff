// Package migrate implements the bakeoff migration runner.
// Design: bakeoff#25 (schema), bakeoff#27 (Phase 1 function signatures).
package migrate

import (
	"fmt"

	"github.com/d5/tengo/v2"
)

// Record wraps a database row during record migration script execution.
// All mutations are tracked so only dirty fields are written back.
type Record struct {
	data      map[string]any
	dirty     map[string]any  // fields explicitly set via setField
	deleted   map[string]bool // fields added to delete-set via deleteField
	rejected  bool
	rejectMsg string
	ignored   bool
	ignoreMsg string
}

// NewRecord constructs a Record from a row map.
func NewRecord(row map[string]any) *Record {
	data := make(map[string]any, len(row))
	for k, v := range row {
		data[k] = v
	}
	return &Record{
		data:    data,
		dirty:   make(map[string]any),
		deleted: make(map[string]bool),
	}
}

// Get returns the current value of field (original or overwritten).
func (r *Record) Get(field string) any {
	if v, ok := r.dirty[field]; ok {
		return v
	}
	return r.data[field]
}

// Set marks field dirty with value.
func (r *Record) Set(field string, value any) {
	r.dirty[field] = value
	delete(r.deleted, field)
}

// Delete adds field to the delete-set; field is omitted on write-back.
func (r *Record) Delete(field string) {
	r.deleted[field] = true
	delete(r.dirty, field)
}

// Reject marks this record as rejected with reason.
func (r *Record) Reject(reason string) {
	r.rejected = true
	r.rejectMsg = reason
}

// Ignore marks this record to be skipped with reason.
func (r *Record) Ignore(reason string) {
	r.ignored = true
	r.ignoreMsg = reason
}

// MergedRow returns the final row to write back.
// Fields in the delete-set are omitted; dirty fields override originals.
func (r *Record) MergedRow() map[string]any {
	out := make(map[string]any, len(r.data))
	for k, v := range r.data {
		if !r.deleted[k] {
			out[k] = v
		}
	}
	for k, v := range r.dirty {
		if !r.deleted[k] {
			out[k] = v
		}
	}
	return out
}

// toTengoMap converts the record's current state to a Tengo map value
// suitable for injection into a Tengo script as the `this` variable.
func (r *Record) toTengoMap() *tengo.Map {
	m := &tengo.Map{Value: make(map[string]tengo.Object)}
	for k, v := range r.data {
		m.Value[k] = goToTengo(v)
	}
	for k, v := range r.dirty {
		if !r.deleted[k] {
			m.Value[k] = goToTengo(v)
		}
	}
	for k := range r.deleted {
		delete(m.Value, k)
	}
	return m
}

// goToTengo converts a Go value to a Tengo object (best-effort).
func goToTengo(v any) tengo.Object {
	if v == nil {
		return tengo.UndefinedValue
	}
	switch val := v.(type) {
	case bool:
		if val {
			return tengo.TrueValue
		}
		return tengo.FalseValue
	case int:
		return &tengo.Int{Value: int64(val)}
	case int32:
		return &tengo.Int{Value: int64(val)}
	case int64:
		return &tengo.Int{Value: val}
	case float32:
		return &tengo.Float{Value: float64(val)}
	case float64:
		return &tengo.Float{Value: val}
	case string:
		return &tengo.String{Value: val}
	case []byte:
		return &tengo.Bytes{Value: val}
	case map[string]any:
		m := &tengo.Map{Value: make(map[string]tengo.Object, len(val))}
		for k, vv := range val {
			m.Value[k] = goToTengo(vv)
		}
		return m
	case []any:
		arr := &tengo.Array{Value: make([]tengo.Object, len(val))}
		for i, vv := range val {
			arr.Value[i] = goToTengo(vv)
		}
		return arr
	default:
		return &tengo.String{Value: fmt.Sprintf("%v", val)}
	}
}

// tengoToGo converts a Tengo object to a Go value.
func tengoToGo(obj tengo.Object) any {
	if obj == nil || obj == tengo.UndefinedValue {
		return nil
	}
	switch v := obj.(type) {
	case *tengo.Bool:
		return !v.IsFalsy()
	case *tengo.Int:
		return v.Value
	case *tengo.Float:
		return v.Value
	case *tengo.String:
		return v.Value
	case *tengo.Bytes:
		return v.Value
	case *tengo.Array:
		out := make([]any, len(v.Value))
		for i, elem := range v.Value {
			out[i] = tengoToGo(elem)
		}
		return out
	case *tengo.Map:
		out := make(map[string]any, len(v.Value))
		for k, vv := range v.Value {
			out[k] = tengoToGo(vv)
		}
		return out
	default:
		return v.String()
	}
}
