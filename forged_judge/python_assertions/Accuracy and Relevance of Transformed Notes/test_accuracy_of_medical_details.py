def test_accuracy_of_medical_details(self):
    medical_notes = self.output.get("output", "")
    self.assertIn(
        "Chief complaint: History of right ductal carcinoma in situ (DCIS).",
        medical_notes,
    )
    self.assertIn(
        "History of present illness: New patient evaluation for continued monitoring of right DCIS.",
        medical_notes,
    )
