# JaxNN MVP - Issues & Proposed Changes

## Critical Bugs (must fix for MVP)

### 1. Local-dir weight loading is broken
**`_resolve_pretrained_source()` in `_builder.py:120-130`** checks for keys `local_dir` and `folder`, but `load_model_config_from_path()` in `_hub.py:129` sets the key as `"file"`. The weights directory is never found.

**Fix:** Add `"file"` to the key lookup in `_resolve_pretrained_source`:
```python
for key in ("file", "local_dir", "folder"):
```

### 2. Windows path mangling in `parse_model_name`
**`_factory.py:16-25`** uses `urlsplit()` which misparses Windows paths like `local-dir:C:\Users\...` — the drive letter `C` becomes the hostname, not part of the path.

**Fix:** Use a simple prefix-check instead of `urlsplit`:
```python
def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    if model_name.startswith('hf-hub:'):
        return 'hf-hub', model_name[len('hf-hub:'):]
    elif model_name.startswith('local-dir:'):
        return 'local-dir', model_name[len('local-dir:'):]
    else:
        model_name = os.path.split(model_name)[-1]
        return None, model_name
```

### 3. Duplicate key check in `_parse_model_cfg`
**`_hub.py:103-104`** checks `label_names` twice but assigns `label_descriptions`:
```python
if "label_names" in cfg:        # ← should be "label_descriptions"
    pretrained_cfg["label_descriptions"] = cfg["label_descriptions"]
```

### 4. `list_models` default `include_tags` is the type `bool` not a boolean
**`_registry.py:168`**: `include_tags: Optional[bool] = bool` — `bool` is truthy, so `include_tags` is always treated as True.

**Fix:** `include_tags: Optional[bool] = None`

## Important Improvements (strongly recommended for MVP)

### 5. `load_checkpoint` is a stub
**`_helpers.py:18-84`** accepts many params (remap, filter_fn, exclude/include patterns, prefix, format detection) but ignores all of them — just calls `load_pretrained(model, pretrained_cfg)`. The `checkpoint_path` argument is never used.

**Fix:** Implement at minimum: load state from `checkpoint_path` using `load_orbax_state_dict`, then apply via `_apply_flat_state_dict`. Other features (remap, filter, patterns) can be deferred.

### 6. Classifier head adaptation is commented out
**`_builder.py:469-483`** — the num_classes mismatch handling is completely commented out. Loading a model with a different num_classes will crash in strict mode.

**Fix:** Uncomment and adapt the classifier head removal logic for Flax key naming.

### 7. No tests
**`tests/__init__.py`** is empty. At minimum, add:
- Test `create_model('resnet34', pretrained=False)` works
- Test `parse_model_name` with all schemes (plain, hf-hub, local-dir)
- Test `load_model_config_from_path` with a dummy config.json
- Test `_resolve_pretrained_source` key resolution
- Test `list_models` / `list_pretrained`

## Minor Issues / Nice-to-haves

### 8. `_rcfg` has wrong `test_input_size` format
**`resnet.py:468`**: `'test_input_size': (3, 288, 288)` is PyTorch CHW format. Flax uses HWC → should be `(288, 288, 3)`.

### 9. `download_cached_file` is fully commented out
**`_hub.py:133-155`** — it relied on torch.hub. Either reimplement with `urllib` + hash verification or remove the dead code.

### 10. `_builder2.py` exists as dead code
Appears to be an incomplete alternative builder. Should be removed or merged.

### 11. `save_for_hf` uses `StandardCheckpointer` while `load` uses `_get_checkpointer`
Inconsistent checkpointer usage between save (`_hub.py:261`) and load (`_builder.py:133`). Should use the same approach.

### 12. `DropPath` uses `nnx.Rngs.params()` incorrectly
**`resnet.py:241`**: `jax.random.bernoulli(nnx.Rngs.params(), ...)` — `Rngs.params()` is an instance method, not a class method. Needs a stored rngs instance.

---

## Suggested Priority Order
1. Fix #1 (local-dir source key) — **breaks local loading entirely**
2. Fix #2 (Windows path parsing) — **breaks local-dir on Windows**
3. Fix #3 (label_descriptions typo)
4. Fix #4 (list_models default)
5. Fix #8 (test_input_size format)
6. Implement #5 (load_checkpoint from path)
7. Implement #6 (classifier head adaptation)
8. Add #7 (basic tests)
