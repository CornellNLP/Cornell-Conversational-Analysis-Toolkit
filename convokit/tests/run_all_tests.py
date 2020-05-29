from unittest import TestLoader, TextTestRunner

if __name__ == "__main__":
    loader = TestLoader()
    tests = loader.discover('.')
    testRunner = TextTestRunner()
    test_results = testRunner.run(tests)

    if len(test_results.errors) > 0:
        exit(1)

