
import os
import subprocess
import uuid

def compare_organ(ct_path, mask1_path, mask2_path, organ, base_output="./comparison_results", base_csv="./results", port="8000"):
    """
    Compare two segmentation masks using vLLM pipeline, safe for multiple calls.

    Args:
        ct_path (str): Path to CT volume (.nii.gz).
        mask1_path (str): Path to segmentation directory for model 1.
        mask2_path (str): Path to segmentation directory for model 2.
        organ (str): Organ name (e.g. "aorta", "pancreas").
        base_output (str): Base folder for projection results.
        base_csv (str): Base folder for CSV results.
        port (int): API server port (vLLM).
    Returns:
        str: 'mask1' or 'mask2' depending on which is judged better.
    """

    # Unique ID for this run
    run_id = uuid.uuid4().hex[:8]
    output_dir = os.path.join(base_output, f"{run_id}", f"{organ}")
    csv_path = os.path.join(base_csv, f"{run_id}", f"{organ}.csv")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(base_csv, f"{run_id}"), exist_ok=True)

    # Step 1. Projection
    result = subprocess.run([
        "python3", "ProjectDatasetFlex_single.py",
        "--ct_good", ct_path,
        "--mask_good", mask1_path,
        "--ct_bad", ct_path,
        "--mask_bad", mask2_path,
        "--output_dir", output_dir
    ], capture_output=True, text=True)
    
    print("STDOUT:\n", result.stdout)
    print("STDERR:\n", result.stderr)
    result.check_returncode()

    # Step 2. Run API
    subprocess.run([
        "python3", "RunAPI_single.py",
        "--path", output_dir,
        "--organ", organ,
        "--csv_path", csv_path,
        "--port", str(port)
    ], check=True)

    # Step 3. Parse result
    better = None
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        row = f.readline().strip().split(",")
        if len(row) >= 2:
            answer = row[1].strip()
            if answer == "0":
                better = "mask1"
            elif answer == "1":
                better = "mask2"
            elif answer == "0.5":
                better = "mask1"
                print("Undecided case (answer=0.5):", ",".join(row))

    return mask1_path if better == "mask1" else mask2_path

print(compare_organ('data/AbdomenAtlasPro_top10/BDMAP_00000001/ct.nii.gz',
'data/AA1.1_dhe23_ccvl40_masked/BDMAP_00000001/segmentations',
'data/AA1.1_dhe23_ccvl40/BDMAP_00000001/segmentations', 'aorta')
)