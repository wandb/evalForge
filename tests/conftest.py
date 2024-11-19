
def pytest_configure(config):
    config.option.vcr_record_mode = 'once' 