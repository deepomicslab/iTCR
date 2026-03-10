"""
Custom sample parser functions for iTCR analysis
MODIFY THESE FUNCTIONS according to your sample naming convention

IMPORTANT INSTRUCTIONS:
======================

1. SAMPLE ID FORMAT:
   Your sample IDs should follow this format:
   - Pre-treatment samples: "patient_id pretreatment" (e.g., "UPN1 pretreatment")
   - Post-treatment samples: "patient_id posttreatment" (e.g., "UPN1 posttreatment")
   
2. SAMPLE MAPPING DICTIONARY FORMAT:
   The create_sample_mapping() function should return a dictionary with this structure:
   {
       "patient_id": {
          the patient information....
       }
   }

3. EXAMPLE:
   Sample IDs in your data: ["UPN1 pretreatment", "UPN1 posttreatment", "UPN4 pretreatment", ...]
   Patient mapping: {"UPN1": {"posttreatment": "3M_CR", "cmv_status": "Positive", ...}, ...}

4. TO CUSTOMIZE:
   - Modify create_sample_mapping() with your patient metadata

"""


def create_sample_mapping():
    """
    Create sample mapping dictionary
    MODIFY THIS FUNCTION according to your sample naming convention
    
    Returns:
    --------
    dict: Mapping of patient IDs to their metadata
    """
    return {
        "UPN1": {"pre": "Pre", "posttreatment": "3M_CR", "cmv_status": "Positive", "3M_response": "CR", "6M_response": "CR"},
        "UPN4": {"pre": "Pre", "posttreatment": "3M_PR", "cmv_status": "Positive", "3M_response": "PR", "6M_response": "Relapsed"},
        "UPN6": {"pre": "Pre", "posttreatment": None, "cmv_status": "Negative", "3M_response": "NR", "6M_response": "NE, off"},
        "UPN8": {"pre": "Pre", "posttreatment": "6M_CR", "cmv_status": "Positive", "3M_response": "CR", "6M_response": "CR"},
        "UPN10": {"pre": "Pre", "posttreatment": "3M_CR", "cmv_status": "Positive", "3M_response": "CR", "6M_response": "CR"},
        "UPN12": {"pre": "Pre", "posttreatment": "3M_CR", "cmv_status": "Negative", "3M_response": "CR", "6M_response": "CR"},
        "UPN13": {"pre": "Pre", "posttreatment": "6M_PR", "cmv_status": "Negative", "3M_response": "PR", "6M_response": "PR"},
        "UPN14": {"pre": "Pre", "posttreatment": "6M_PR", "cmv_status": "Negative", "3M_response": "NR", "6M_response": "PR"},
        "UPN15": {"pre": "Pre", "posttreatment": "6M_NR", "cmv_status": "Positive", "3M_response": "NR", "6M_response": "NR"},
        "UPN17": {"pre": "Pre", "posttreatment": "6M_NR", "cmv_status": "Positive", "3M_response": "NR", "6M_response": "NR, off"},
        "UPN18": {"pre": "Pre", "posttreatment": "3M_PR", "cmv_status": "Negative", "3M_response": "PR", "6M_response": "Relapsed"},
        "UPN19": {"pre": "Pre", "posttreatment": "3M_PR", "cmv_status": "Negative", "3M_response": "PR", "6M_response": "PR"},
        "UPN24": {"pre": "Pre", "posttreatment": "3M_CR", "cmv_status": "Positive", "3M_response": "CR", "6M_response": "CR"}
    }



