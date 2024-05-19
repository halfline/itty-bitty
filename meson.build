project ('itty-bitty', 'c', default_options : ['c_std=c11'])

source_files = [
        'src/itty-bit-string.c',
        'src/main.c'
]

executable ('itty-bitty',
            sources: source_files,
            install: true)

test_source_files = [
        'src/tests/test-itty-bit-string.c'
]

test_executable = executable ('test-itty-bit-string',
                              sources: test_source_files,
                              include_directories: include_directories ('src'))

test ('itty-bit-string-test', test_executable)

