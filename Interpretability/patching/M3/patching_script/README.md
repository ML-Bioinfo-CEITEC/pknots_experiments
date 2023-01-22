# Patching Scripts

Latest version: [calculate_moving_patch_results.py](./calculate_moving_patch_results.py)

Expects the input CSV file to be in this exact format:

```
seq_id,sequence,global_metric_value,domain_architecture,interpro,max_knot_topology,seq_length,label,family,knot_start,knot_end,knot_len,core_percentage
```

Usage: 

```
python3 calculate_moving_patch_results.py --patch_sizes 10 50 80 200 --input_path <INPUT_PATH> --output_path <OUTPUT_PATH>
```

Arguments (required are **bold**): 

| Short | Long argument | Description | Default |
|-|-|-|-|
|`-ps`|`--patch_sizes`|List of patch sizes given as integers delimeted by a whitespace.|`10 50 80 200`|
|`-fl`|`--fix_length`|Use if input sequences are supposed to be padded to length 500.|-|
|`-i`|`--input_path`|**Input CSV path**|-|
|`-si`|`--start_index`|Start index of first processed sequence from CSV file|`1`|
|`-a`|`--append`|Append results to the output file content (default is overwriting)|-|
|`-o`|`--output_path`|**Output CSV path**|-|


