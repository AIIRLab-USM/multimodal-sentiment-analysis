from src.model_training import bert_trainer, vit_trainer, multimodal_trainer
from src.model_evaluation import text_evaluation, image_evaluation, multimodal_evaluation, mcnemar_test, style_analysis
from src.data_preprocessing import process_artemis, caption_generation
from src import data_visualization

"""
Script for project-ease-of-use
The following script executes code within this repository to do the following

    1.  Customize the ArtEmis dataset
    
    2.  Generate captions for the ArtEmis dataset
    
        (The resulting data for steps one and two may be found in the 'data' directory
    
    3.  Train a unimodal text and image models for multi-class sentiment classification
        The architecture of these models may be found in 'classification_models.py'
        These models are trained on the dataset dervied from step two of this script
        
    4.  Evaluate models as described in step three
        Final test-results and metrics may be found in the 'model_evaluation' directory
        
If a user wishes to execute one or more of these steps in isolation, you may do so by commenting out the other tasks

Author: Clayton Durepos
Version: 05.04.2025
Contact: clayton.durepos@maine.edu
"""

if __name__ == '__main__':
    # Process original data
    process_artemis.main()

    # # Generate captions for final dataset
    # # NOTICE: This step takes especially long
    caption_generation.main()

    # Train text-classification model
    bert_trainer.main()

    # Train image-classification model
    # NOTICE: This step takes especially long
    vit_trainer.main()

    # Train multi-modal classification model
    # NOTICE: This step takes especially long
    multimodal_trainer.main()

    # Evaluate text-model
    text_evaluation.main()

    # Evaluate image-model
    image_evaluation.main()

    # Evaluate multi-modal model
    multimodal_evaluation.main()

    # Bootstrap testing
    mcnemar_test.main()

    # Generate confusion matrices for each modality
    # Generate distribution plots
    data_visualization.main()

    # Per-style analysis for multimodal model
    style_analysis.main()