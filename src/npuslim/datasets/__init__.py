from npuslim.core import DatasetRegistry

DatasetRegistry.register_lazy("C4", ".c4_dataset", "C4Dataset")
DatasetRegistry.register_lazy("Text", ".text_dataset", ["TextDataset", "text"])
