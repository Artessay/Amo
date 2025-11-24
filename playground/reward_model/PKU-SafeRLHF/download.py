import json
import datasets

def process_dataset(mode: str):
    part_dataset = dataset[mode]
    data_list = part_dataset.to_list()

    def build_item(data):
        return {
            "prompt": data["prompt"],
            "response_0": data["response_0"],
            "response_1": data["response_1"],
            # "is_response_0_safe": data["is_response_0_safe"],
            # "is_response_1_safe": data["is_response_1_safe"],
            "better_response_id": data["better_response_id"],
            "safer_response_id": data["safer_response_id"],
        }

    data_list = [
        build_item(data) 
        for data in data_list
    ]
    
    with open(f"{mode}.json", "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=4)

if __name__ == "__main__":
    dataset = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF")
    process_dataset("train")
    process_dataset("test")
