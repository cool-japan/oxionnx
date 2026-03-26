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

## Usage

```toml
[dependencies]
oxionnx-proto = "0.1.0"
```

```rust
use oxionnx_proto::{load, parse_model};

let bytes = std::fs::read("model.onnx").expect("read model");
let (graph, weights) = load(&bytes).expect("parse model");

println!("Graph has {} nodes", graph.nodes.len());
println!("Loaded {} weight tensors", weights.len());
```

### Streaming parser for large models

```rust
use oxionnx_proto::{parse_streaming, ParseEvent};

let bytes = std::fs::read("large_model.onnx").expect("read model");
for event in parse_streaming(&bytes).expect("parse") {
    match event {
        ParseEvent::Node(node) => println!("op: {:?}", node.op_type),
        ParseEvent::Initializer(name, _tensor) => println!("weight: {name}"),
        _ => {}
    }
}
```

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `mmap`  | No      | Enables memory-mapped file loading via `memmap2` for reduced memory usage on large models. |

## Part of [oxionnx](https://github.com/cool-japan/oxionnx)

A Pure Rust ONNX inference engine.

## License

Apache-2.0
