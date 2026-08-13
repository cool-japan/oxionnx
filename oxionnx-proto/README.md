# oxionnx-proto

ONNX protobuf parser for OxiONNX -- zero-dependency wire format decoder.

This crate reads `.onnx` model files (ONNX protobuf format) and converts them
into the `oxionnx_core::Graph` representation. It includes a hand-written protobuf
decoder with no dependency on `prost` or `protobuf` crates, keeping the dependency
tree minimal and fully Pure Rust.

## Key Functionality

- **`load(bytes)`** -- Parse an in-memory ONNX model and return a `(Graph, HashMap<String, Tensor>)` of the computation graph and weight tensors.
- **`load_with_path(bytes, base_dir)`** -- Same as `load`, but resolves external data files relative to the given path.
- **`parse_model(bytes)`** -- Low-level protobuf parser returning the raw `ModelProto`.
- **`StreamingParser`** / **`parse_streaming`** -- Memory-efficient streaming parser that emits `ParseEvent`s, useful for large models.
- **`parse_with_weight_filter`** -- Streaming parser that selectively loads weights matching a filter predicate.
- **Schema validation** -- `validate_schemas` checks nodes against built-in `OpSchema` definitions for input/output arity and attribute correctness.
- **Supported opset range** -- Opsets 7 through 21.
- **ONNX local function inlining** -- `ModelProto.functions` (function calls) are resolved by inlining into the graph before `load`, `parse_streaming`, and `parse_with_weight_filter` all return; a model that calls a local function loads identically through every entry point. (Before 0.1.5 only `load`'s eager path did this -- the streaming path failed such models with `UnsupportedOp`.)

## Hardening against untrusted models

`.onnx` files are untrusted input by construction, so both the eager and
streaming parsers are defensive by default:

- Length-delimited reads are bounds-checked (`pos + len` against the buffer)
  everywhere -- including a previously fully-unguarded slice in the
  streaming reader -- instead of panicking on truncated or malformed bytes.
- Nested subgraph attributes are depth-limited instead of recursing without
  bound, closing a stack-overflow-on-a-crafted-model path.
- Lengths read from the model are clamped before sizing an allocation, so a
  hostile length can no longer OOM-abort the process.
- `TensorProto` typed-data fields use the correct field numbers and element
  encodings; non-packed (unpacked) repeated numeric fields are read instead
  of silently dropped; protobuf group wire types are skipped instead of
  aborting the parse; the varint decoder no longer truncates bits above 64
  on a 10-byte varint.
- `GraphProto.value_info` and `AttributeProto.tensors`/`graphs`/
  `ref_attr_name` are parsed instead of silently dropped.
  `GraphProto.sparse_initializer` and `AttributeProto.sparse_tensor`/
  `type_proto` are not yet supported -- rather than silently dropping the
  data (which for `sparse_initializer` would leave the graph referencing a
  tensor that never materializes) or misparsing it, a model that uses them
  fails to load with a diagnosable error naming the field.
- Weight decode covers uint8/int8/bool/double/bfloat16 initializers without
  zero-filling, and negative or overflowing `dims` are rejected with a typed
  error instead of being cast to a huge `usize`.
- External-data `location` paths are canonicalized and checked against the
  model directory (sandboxed against path traversal and absolute-path
  reads), with checked offset/length arithmetic instead of panicking on a
  reversed or overflowing range.
- `AttributeProto.strings` (`repeated bytes`, field 9) is read verbatim
  instead of being re-parsed as a nested protobuf message -- the old
  behavior corrupted ONNX-ML `TreeEnsemble` models whose `nodes_modes`
  values (e.g. `"BRANCH_GTE"`) happen to decode as a plausible tag+length
  prefix, raising a spurious `length-delimited: EOF` error (issue #3,
  regression-tested).

## Usage

```toml
[dependencies]
oxionnx-proto = "0.1.8"
```

```rust
use oxionnx_proto::load;

let bytes = std::fs::read("model.onnx").expect("read model");
let (graph, weights) = load(&bytes).expect("parse model");

println!("Graph has {} nodes", graph.nodes.len());
println!("Loaded {} weight tensors", weights.len());
```

### Streaming parser for large models

`StreamingParser` reads incrementally from any `Read` source and calls back
per event -- nodes, weights, and metadata are never all buffered in memory
at once, unlike the convenience `parse_streaming`/`parse_with_weight_filter`
wrappers (which read the whole model but return an already-assembled graph
and weight map for callers that don't need the low-level event stream):

```rust
use std::io::Cursor;
use oxionnx_proto::{ParseEvent, StreamingParser};

let bytes = std::fs::read("large_model.onnx").expect("read model");
let mut parser = StreamingParser::new(Cursor::new(bytes));
parser
    .parse(|event| {
        match event {
            ParseEvent::Node(node) => println!("op: {:?}", node.op_type),
            ParseEvent::Weight { name, tensor } => {
                println!("weight: {name} ({} elements)", tensor.data.len());
            }
            _ => {}
        }
        Ok(())
    })
    .expect("parse");
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `mmap`  | No      | Enables memory-mapped file loading via `memmap2` for reduced memory usage on large models. |

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
