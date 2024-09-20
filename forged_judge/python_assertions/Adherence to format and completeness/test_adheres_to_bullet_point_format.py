def test_adheres_to_bullet_point_format(self):
    required_keys = ['output']
    for key in required_keys:
        self.assertIn(key, self.output)
    sections = [s.strip() for s in self.output['output'].split('\n') if s.strip()]
    for section in sections:
        self.assertTrue(section.startswith('\u2022'), f'Section does not start with a bullet point: {section}')