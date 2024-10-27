# import os
# import csv
# import pandas as pd

# '''
# Entities:
# 1. employerName
# 2. employerAddressStreet_name
# 3. employerAddressCity
# 4. employerAddressState
# 5. employerAddressZip
# 6. einEmployerIdentificationNumber
# 7. employeeName
# 8. ssnOfEmployee
# 9. box1WagesTipsAndOtherCompensations
# 10. box2FederalIncomeTaxWithheld
# 11. box3SocialSecurityWages
# 12. box4SocialSecurityTaxWithheld
# 13. box16StateWagesTips
# 14. box17StateIncomeTax
# 15. taxYear
# '''



# '''
# Description: The fuction yields the standard precision, recall and f1 score metrics

# arguments:
#     TP -> int
#     FP -> int
#     FN -> int

# returns: float, float, float
# '''
# def performance(TP, FP, FN):
    
#     if (TP+FP) == 0:
#         precision = "NaN"
#     else:
#         precision = TP/float((TP+FP))
        
#     if (TP+FN) == 0:
#         recall = "NaN"
#     else:
#         recall = TP/float((TP+FN))
    
#     if (recall!="NaN") and (precision!="NaN"):
#         f1_score = (2.0*precision*recall)/(precision+recall)
#     else:
#         f1_score = "NaN"
    
#     return precision, recall, f1_score
    
    
    
    
# '''
# Description: The fuction yields a dataframe containing entity-wise performance metrics

# arguments:
#     true_labels -> list
#     pred_labels -> lisyt
    
# returns: pandas dataframe
# '''
# def get_dataset_metrics(true_labels, pred_labels):
    
#     metrics_dict = dict()
    
#     for true_label, pred_label in zip(true_labels, pred_labels):
#         if true_label not in metrics_dict:
#             metrics_dict[true_label] = {"TP":0, "FP":0, "FN":0, "Support":0}
        
#         if true_label != "OTHER":
#             metrics_dict[true_label]["Support"] += 1
            
#             if true_label == pred_label:
#                 metrics_dict[true_label]["TP"] += 1
            
#             elif pred_label == "OTHER":
#                 metrics_dict[true_label]["FN"] += 1
            
#         else:
#             if pred_label != "OTHER":
#                 metrics_dict[pred_label]["FP"] += 1
           
#     df = pd.DataFrame()
    
#     for field in metrics_dict:
#         precision, recall, f1_score = performance(metrics_dict[field]["TP"], metrics_dict[field]["FP"], metrics_dict[field]["FN"])
#         support = metrics_dict[field]["Support"]
        
#         if field != "OTHER":
#             temp_df = pd.DataFrame([[precision, recall, f1_score, support]], columns=["Precision", "Recall", "F1-Score", "Support"], index=[field])
#             df = df.append(temp_df)
    
#     return df




# '''
# Description: The fuction yields a dataframe containing entity-wise performance metrics for a single document
# (make sure the doc id is the same)

# arguments:
#     doc_true -> tsv file with with labels in the last column (8 th column (1-indexed))
#     doc_pred -> tsv file with labels in the last column (8 th column (1-indexed)), as predicted by the model
    
# returns: list, list
# '''
# def get_doc_labels(doc_true, doc_pred):

#     true_labels = [row[-1] for row in csv.reader(open(doc_true, "r"))]
#     pred_labels = [row[-1] for row in csv.reader(open(doc_pred, "r"))]

#     return true_labels, pred_labels



# '''
# Description: The fuction yields a dataframe containing entity-wise performance metrics for all documents
# (make sure the doc ids are the same in both the paths)

# arguments:
#     doc_true -> string (directory containing the ground truth tsv files)
#     doc_pred -> string (directory containing the predicted tsv files)
#     save -> bool (saves the metrics file in your working directory)
# returns: pandas dataframe
# '''
# def get_dataset_labels(true_path, pred_path, save=False):
    
#     y_true, y_pred = [], []
    
#     for true_file in os.listdir(true_path):
#         for pred_file in os.listdir(pred_path):
#             if (".tsv" in true_file) and (".tsv" in pred_file):
#                 if true_file == pred_file:
                    
#                     true_file, pred_file = f"{true_path}/{true_file}", f"{pred_path}/{pred_file}"
#                     true_labels, pred_labels = get_doc_labels(true_file, pred_file)
                    
#                     y_true.extend(true_labels)
#                     y_pred.extend(pred_labels)
            
#     df = get_dataset_metrics(y_true, y_pred)
#     print(df)
#     if save == True:
#         df.to_csv("eval_metrics.tsv")



# if __name__ == "__main__":
    
#     # template to run your own evaluation

#     doc_true = f"{os.getcwd()}/train/boxes_transcripts_labels"
#     doc_pred = f"{os.getcwd()}/train/boxes_transcripts_labels"

#     get_dataset_labels(doc_true, doc_pred, save=False)

        
        
# import json
# import os
# from pathlib import Path
# import logging
# from typing import Dict, Any, List, Tuple
# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support
# from w2_extractor import W2FormExtractor  # Import from your tesseract.py file

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class W2EvaluationMetrics:
#     def __init__(self):
#         self.total_samples = 0
#         self.successful_extractions = 0
#         self.field_matches = {}
#         self.field_errors = {}
        
#     def update(self, ground_truth: dict, prediction: dict) -> None:
#         """Update metrics with a single sample comparison."""
#         self.total_samples += 1
#         if prediction is not None:
#             self.successful_extractions += 1
            
#             # Compare each field
#             for field in ground_truth:
#                 if field not in self.field_matches:
#                     self.field_matches[field] = 0
#                     self.field_errors[field] = []
                
#                 gt_value = self._normalize_value(ground_truth[field])
#                 pred_value = self._normalize_value(prediction.get(field))
                
#                 if gt_value == pred_value:
#                     self.field_matches[field] += 1
#                 else:
#                     self.field_errors[field].append((gt_value, pred_value))
    
#     def _normalize_value(self, value: Any) -> str:
#         """Normalize values for comparison."""
#         if value is None:
#             return ""
#         # Remove spaces, convert to lowercase, and remove special characters
#         return str(value).lower().replace(" ", "").replace("$", "").replace(",", "")
    
#     def get_metrics(self) -> Dict[str, Any]:
#         """Calculate and return evaluation metrics."""
#         metrics = {
#             "total_samples": self.total_samples,
#             "successful_extractions": self.successful_extractions,
#             "extraction_success_rate": self.successful_extractions / self.total_samples if self.total_samples > 0 else 0,
#             "field_accuracy": {}
#         }
        
#         # Calculate per-field accuracy
#         for field in self.field_matches:
#             accuracy = self.field_matches[field] / self.total_samples if self.total_samples > 0 else 0
#             metrics["field_accuracy"][field] = {
#                 "accuracy": accuracy,
#                 "correct_matches": self.field_matches[field],
#                 "total_samples": self.total_samples
#             }
        
#         return metrics

# def load_ground_truth(annotations_dir: str) -> Dict[str, Dict]:
#     """Load ground truth annotations from JSON files."""
#     ground_truth = {}
#     for file_path in Path(annotations_dir).glob("*.json"):
#         try:
#             with open(file_path, 'r') as f:
#                 annotation = json.load(f)
#                 ground_truth[file_path.stem] = annotation
#         except Exception as e:
#             logger.error(f"Error loading annotation file {file_path}: {str(e)}")
#     return ground_truth

# def evaluate_w2_extraction(
#     image_dir: str,
#     annotations_dir: str,
#     results_dir: str,
#     tesseract_path: str = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
# ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
#     """
#     Evaluate W2 form extraction against ground truth annotations.
    
#     Args:
#         image_dir: Directory containing W2 form images
#         annotations_dir: Directory containing ground truth annotations
#         results_dir: Directory to save evaluation results
#         tesseract_path: Path to Tesseract executable
    
#     Returns:
#         Tuple of (metrics_dict, detailed_results)
#     """
#     try:
#         # Initialize W2 Form Extractor
#         extractor = W2FormExtractor(tesseract_path=tesseract_path)
        
#         # Load ground truth annotations
#         ground_truth = load_ground_truth(annotations_dir)
#         if not ground_truth:
#             raise ValueError("No ground truth annotations found")
        
#         # Initialize metrics
#         metrics = W2EvaluationMetrics()
#         detailed_results = []
        
#         # Process each image
#         for image_file in Path(image_dir).glob("*.[pj][np][gf]*"):
#             image_id = image_file.stem
            
#             if image_id not in ground_truth:
#                 logger.warning(f"No ground truth found for image: {image_id}")
#                 continue
            
#             try:
#                 # Extract data from image
#                 extracted_data = extractor.extract_form_data(str(image_file))
                
#                 # Update metrics
#                 metrics.update(ground_truth[image_id], extracted_data)
                
#                 # Store detailed result
#                 result = {
#                     "image_id": image_id,
#                     "ground_truth": ground_truth[image_id],
#                     "prediction": extracted_data,
#                     "success": True
#                 }
#             except Exception as e:
#                 logger.error(f"Error processing {image_file}: {str(e)}")
#                 result = {
#                     "image_id": image_id,
#                     "ground_truth": ground_truth[image_id],
#                     "prediction": None,
#                     "success": False,
#                     "error": str(e)
#                 }
            
#             detailed_results.append(result)
        
#         # Calculate final metrics
#         final_metrics = metrics.get_metrics()
        
#         # Save results
#         os.makedirs(results_dir, exist_ok=True)
        
#         with open(os.path.join(results_dir, "evaluation_metrics.json"), 'w') as f:
#             json.dump(final_metrics, f, indent=2)
        
#         with open(os.path.join(results_dir, "detailed_results.json"), 'w') as f:
#             json.dump(detailed_results, f, indent=2)
        
#         return final_metrics, detailed_results
    
#     except Exception as e:
#         logger.error(f"Evaluation failed: {str(e)}")
#         raise

# def print_evaluation_report(metrics: Dict[str, Any], detailed_results: List[Dict[str, Any]]) -> None:
#     """Print a formatted evaluation report."""
#     print("\n=== W2 Form Extraction Evaluation Report ===")
#     print(f"\nTotal samples processed: {metrics['total_samples']}")
#     print(f"Successful extractions: {metrics['successful_extractions']}")
#     print(f"Extraction success rate: {metrics['extraction_success_rate']:.2%}")
    
#     print("\nPer-field Accuracy:")
#     for field, field_metrics in metrics['field_accuracy'].items():
#         print(f"- {field}: {field_metrics['accuracy']:.2%} "
#               f"({field_metrics['correct_matches']}/{field_metrics['total_samples']})")
    
#     print("\nFailed Extractions:")
#     failed_count = sum(1 for result in detailed_results if not result['success'])
#     if failed_count > 0:
#         for result in detailed_results:
#             if not result['success']:
#                 print(f"- {result['image_id']}: {result.get('error', 'Unknown error')}")
#     else:
#         print("None")

# def main():
#     """Main function to run evaluation."""
#     try:
#         # Define paths
#         base_dir = Path(__file__).parent
#         dataset_dir = base_dir / "dataset"
        
#         # Evaluation settings
#         settings = {
#             "image_dir": str(dataset_dir / "train" / "images"),
#             "annotations_dir": str(dataset_dir / "train" / "annotations"),
#             "results_dir": str(base_dir / "evaluation_results"),
#             "tesseract_path": r"D:\Program Files\Tesseract-OCR\tesseract.exe"
#         }
        
#         # Run evaluation
#         logger.info("Starting W2 form extraction evaluation...")
#         metrics, detailed_results = evaluate_w2_extraction(**settings)
        
#         # Print report
#         print_evaluation_report(metrics, detailed_results)
        
#         logger.info(f"Evaluation complete. Results saved to: {settings['results_dir']}")
        
#     except Exception as e:
#         logger.error(f"Evaluation failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()

import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class W2EvaluationMetrics:
    def __init__(self):
        self.total_samples = 0
        self.successful_extractions = 0
        self.field_matches = {}
        self.field_errors = {}
        
    def update(self, ground_truth: dict, prediction: dict) -> None:
        """Update metrics with a single sample comparison."""
        self.total_samples += 1
        if prediction is not None:
            self.successful_extractions += 1
            
            # Compare each field
            for field in ground_truth:
                if field not in self.field_matches:
                    self.field_matches[field] = 0
                    self.field_errors[field] = []
                
                gt_value = self._normalize_value(ground_truth[field])
                pred_value = self._normalize_value(prediction.get(field))
                
                if gt_value == pred_value:
                    self.field_matches[field] += 1
                else:
                    self.field_errors[field].append((gt_value, pred_value))
    
    def _normalize_value(self, value: Any) -> str:
        """Normalize values for comparison."""
        if value is None:
            return ""
        return str(value).lower().replace(" ", "").replace("$", "").replace(",", "")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate and return evaluation metrics."""
        metrics = {
            "total_samples": self.total_samples,
            "successful_extractions": self.successful_extractions,
            "extraction_success_rate": self.successful_extractions / self.total_samples if self.total_samples > 0 else 0,
            "field_accuracy": {}
        }
        
        for field in self.field_matches:
            accuracy = self.field_matches[field] / self.total_samples if self.total_samples > 0 else 0
            metrics["field_accuracy"][field] = {
                "accuracy": accuracy,
                "correct_matches": self.field_matches[field],
                "total_samples": self.total_samples
            }
        
        return metrics

def load_ground_truth(transcripts_dir: str) -> Dict[str, Dict]:
    """Load ground truth from TSV files."""
    ground_truth = {}
    import csv
    
    # Look for TSV files
    for file_path in Path(transcripts_dir).glob("*.tsv"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read TSV file
                tsv_reader = csv.DictReader(f, delimiter='\t')
                
                # Convert TSV data to dictionary
                data = {}
                for row in tsv_reader:
                    # Assuming the TSV has columns for field name and value
                    # Adjust these column names based on your actual TSV structure
                    if 'field' in row and 'value' in row:
                        data[row['field']] = row['value']
                    elif len(row) == 2:  # If no headers, assume first column is field and second is value
                        field, value = row.values()
                        data[field] = value
                
                if data:  # Only add if we parsed some data
                    # Use the file name without extension as the key
                    key = file_path.stem
                    ground_truth[key] = data
                    logger.info(f"Successfully loaded ground truth for: {key}")
        except Exception as e:
            logger.error(f"Error loading ground truth file {file_path}: {str(e)}")
            
            # Print the first few lines of the file for debugging
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    logger.info(f"First few lines of {file_path.name}:")
                    for i, line in enumerate(f):
                        if i < 5:  # Print first 5 lines
                            logger.info(f"  {line.strip()}")
                        else:
                            break
            except Exception as read_error:
                logger.error(f"Could not read file for debugging: {str(read_error)}")
    
    if not ground_truth:
        logger.warning("No ground truth data could be loaded from TSV files.")
        logger.info(f"Contents of directory {transcripts_dir}:")
        for file_path in Path(transcripts_dir).iterdir():
            logger.info(f"  {file_path.name}")
        
        # Try to read one TSV file to understand its structure
        tsv_files = list(Path(transcripts_dir).glob("*.tsv"))
        if tsv_files:
            sample_file = tsv_files[0]
            logger.info(f"\nAttempting to read sample TSV file: {sample_file.name}")
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    logger.info("File contents:")
                    logger.info(f.read())
            except Exception as e:
                logger.error(f"Error reading sample file: {str(e)}")
    
    return ground_truth

def evaluate_w2_extraction(
    image_dir: str,
    annotations_dir: str,
    results_dir: str,
    tesseract_path: str = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate W2 form extraction against ground truth annotations."""
    try:
        # Initialize W2 Form Extractor
        from w2_extractor import W2FormExtractor
        extractor = W2FormExtractor(tesseract_path=tesseract_path)
        
        # Load ground truth annotations
        ground_truth = load_ground_truth(annotations_dir)
        if not ground_truth:
            raise ValueError("No ground truth annotations could be loaded. Check the annotation files format.")
        
        # Initialize metrics
        metrics = W2EvaluationMetrics()
        detailed_results = []
        
        # Process each image
        for image_file in Path(image_dir).glob("*.[pj][np][gf]*"):
            image_id = image_file.stem
            logger.info(f"Processing image: {image_id}")
            
            try:
                # Extract data from image
                extracted_data = extractor.extract_form_data(str(image_file))
                
                # Get corresponding ground truth
                gt_data = ground_truth.get(image_id)
                if gt_data is None:
                    logger.warning(f"No ground truth found for image: {image_id}")
                    continue
                
                # Update metrics
                metrics.update(gt_data, extracted_data)
                
                # Store detailed result
                result = {
                    "image_id": image_id,
                    "ground_truth": gt_data,
                    "prediction": extracted_data,
                    "success": extracted_data is not None
                }
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {str(e)}")
                result = {
                    "image_id": image_id,
                    "ground_truth": ground_truth.get(image_id),
                    "prediction": None,
                    "success": False,
                    "error": str(e)
                }
            
            detailed_results.append(result)
        
        # Calculate final metrics
        final_metrics = metrics.get_metrics()
        
        # Save results
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, "evaluation_metrics.json"), 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        with open(os.path.join(results_dir, "detailed_results.json"), 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return final_metrics, detailed_results
    
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def print_evaluation_report(metrics: Dict[str, Any], detailed_results: List[Dict[str, Any]]) -> None:
    """Print a formatted evaluation report."""
    print("\n=== W2 Form Extraction Evaluation Report ===")
    print(f"\nTotal Samples: {metrics['total_samples']}")
    print(f"Successful Extractions: {metrics['successful_extractions']}")
    print(f"Extraction Success Rate: {metrics['extraction_success_rate']:.2%}")
    
    print("\nField-level Accuracy:")
    for field, stats in metrics['field_accuracy'].items():
        print(f"  {field}:")
        print(f"    Accuracy: {stats['accuracy']:.2%}")
        print(f"    Correct Matches: {stats['correct_matches']}/{stats['total_samples']}")

def main():
    """Main function to run evaluation."""
    try:
        # Define paths based on your directory structure
        from pathlib import Path

        # Define the base directory

        base_dir = Path("C:\\Users\\shiva\\Downloads\\dataset")

        # Configure settings
        settings = {
            "image_dir": str(base_dir / "train" / "images"),
            "annotations_dir": str(base_dir / "train" / "boxes_transcripts_labels"),
            "results_dir": str(base_dir / "evaluation_results"),
            "tesseract_path": str(base_dir / "tesseract-5.4.1" / "tesseract.exe")
        }

        
        # Log settings
        logger.info("Evaluation settings:")
        for key, value in settings.items():
            logger.info(f"  {key}: {value}")
            if key in ["image_dir", "annotations_dir"]:
                path = Path(value)
                if not path.exists():
                    logger.error(f"Directory does not exist: {value}")
                else:
                    logger.info(f"  - Directory exists and contains {len(list(path.glob('*')))} files")
        
        # Run evaluation
        logger.info("Starting W2 form extraction evaluation...")
        metrics, detailed_results = evaluate_w2_extraction(**settings)
        
        # Print report
        print_evaluation_report(metrics, detailed_results)
        
        logger.info(f"Evaluation complete. Results saved to: {settings['results_dir']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()