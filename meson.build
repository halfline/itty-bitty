project ('itty-bitty', 'c', default_options : ['c_std=c11'])
add_project_arguments('-D_GNU_SOURCE', language: 'c')

library_sources = [
        'src/itty-bit-string.c',
        'src/itty-bit-string-list.c',
        'src/itty-bit-string-map.c',
        'src/itty-manager.c',
        'src/itty-network.c',
        'src/itty-vocabulary.c',
        'src/itty-work-queue.c',
]

libittybitty = library('ittybitty',
                       sources: library_sources,
                       include_directories: include_directories('src'))

executable ('itty-bitty',
            sources: 'src/main.c',
            link_with: libittybitty,
            install: true)

test_sources = [
        'src/tests/test-itty-bit-string.c',
        'src/tests/test-itty-bit-string-list.c',
        'src/tests/test-itty-bit-string-map.c',
        'src/tests/test-itty-manager.c',
        'src/tests/test-itty-vocabulary.c',
        'src/tests/test-itty-work-queue.c'
]

foreach test_source : test_sources
    test_name = test_source.split('/')[-1].split('.')[0]
    test_executable = executable(test_name,
                                 sources: [test_source],
                                 include_directories: include_directories('src'),
                                 link_with: libittybitty)
    test(test_name + '-test', test_executable)
endforeach
