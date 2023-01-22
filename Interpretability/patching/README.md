# Sequence Patching

For each sequence that is knotted, we tried to locate the knot core. The idea is simple: if a sequence is 
knotted, the model decided based on some part(s) of the sequence. If that particular part of the sequence 
is masked (by a patch made up of 'X' characters) the prediction score should drop. 

Patching approach consists of these steps:

1. For each sequence generate it's patched versions in such a way that the patch of size `patch_size`
moves from left to right:

```
Example for patch_size=5:
AAAAAAAAAAAAAAAAAAAAAAAAAA
XXXXXAAAAAAAAAAAAAAAAAAAAA
AXXXXXAAAAAAAAAAAAAAAAAAAA
AAXXXXXAAAAAAAAAAAAAAAAAAA
AAAXXXXXAAAAAAAAAAAAAAAAAA
AAAAXXXXXAAAAAAAAAAAAAAAAA
...
AAAAAAAAAAAAAAAAAAAAAXXXXX
```

2. Put all patched versions of the sequence as an input to the model.

3. Take the overall minimum of the patched versions => this part of the sequence is important for the
decision that the sequence is knotted (it could be the exact part of the sequence that is responsible
for the knotted core).

### M1, M2 

- Both models take data as a HF Dataset - the script that prepares patched sequences for both these 
models is [generate_hf_patched_dataset.ipynb](./generate_hf_patched_dataset.ipynb)

- HF Dataset preparation and model prediction is divided into separate Jupyter notebooks. 

### M3

- All parts of the approach are implemented in the latest version of the patching script. 