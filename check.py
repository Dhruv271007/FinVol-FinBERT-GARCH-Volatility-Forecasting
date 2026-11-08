"""Optional GPU check; safe on machines without CUDA."""
def main():
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

if __name__ == "__main__":
    main()
