import os
import pandas as pd
import numpy as np

def analyze_confusion():
    misclass_file = "scratch/misclassifications.csv"
    if not os.path.exists(misclass_file):
        print(f"Error: Run evaluate_errors.py first to generate {misclass_file}")
        return

    df = pd.read_csv(misclass_file)
    
    # Group confusion occurrences
    confusion_pairs = df.groupby(["Actual", "Predicted"]).size().reset_index(name="Count")
    confusion_pairs = confusion_pairs.sort_values(by="Count", ascending=False)
    
    print("\n" + "="*50)
    print(" TOP 5 MOST CONFUSED CLASS PAIRS")
    print("="*50)
    for i, row in confusion_pairs.head(5).iterrows():
        print(f"  {row['Actual']:<15s} misclassified as {row['Predicted']:<15s} : {row['Count']} times")
    
    # Query specific Plastic vs Glass confusion
    plastic_as_glass = df[(df["Actual"] == "Plastic") & (df["Predicted"] == "Glass")]
    glass_as_plastic = df[(df["Actual"] == "Glass") & (df["Predicted"] == "Plastic")]
    
    print("\n" + "="*50)
    print(" TARGETED ANALYSIS: PLASTIC VS. GLASS")
    print("="*50)
    print(f"Plastic images classified as Glass: {len(plastic_as_glass)}")
    print(f"Glass images classified as Plastic: {len(glass_as_plastic)}")
    
    if len(plastic_as_glass) > 0:
        print("\nExamples of Plastic classified as Glass:")
        for idx, row in plastic_as_glass.head(3).iterrows():
            print(f"  File: {os.path.basename(row['File'])}")
            print(f"    Confidence for Glass:   {row['Prob_Predicted']*100:.2f}%")
            print(f"    Confidence for Plastic: {row['Prob_Actual']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    analyze_confusion()
