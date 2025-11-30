import os
from src.common import config

timit_root = config.TIMIT_ROOT

phonemes = set()

for split in ["TRAIN", "TEST"]:
    print("\n\nTRAIN:")
    for dr in os.listdir(os.path.join(timit_root, split)):
        print("\nDR:", dr)        
        dr_path = os.path.join(timit_root, split, dr)
        if not os.path.isdir(dr_path):
            continue

        for spkr in os.listdir(dr_path):
            print("Speaker:", spkr)        
            spkr_path = os.path.join(dr_path, spkr)
            if not os.path.isdir(spkr_path):
                continue

            for fname in os.listdir(spkr_path):
                if fname.endswith(".PHN") or fname.endswith(".phn"):
                    with open(os.path.join(spkr_path, fname)) as f:
                        for line in f:
                            _, _, phn = line.strip().split()
                            phonemes.add(phn)

print(sorted(phonemes))
print("Total:", len(phonemes))
