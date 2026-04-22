project ('itty-bitty', 'c', default_options : ['c_std=c11'])
add_project_arguments('-D_GNU_SOURCE', language: 'c')
cc = meson.get_compiler('c')
math_dep = cc.find_library('m', required: false)

library_sources = [
        'src/itty-affinity.c',
        'src/itty-bit-string.c',
        'src/itty-bit-string-list.c',
        'src/itty-bit-string-map.c',
        'src/itty-exec-buffer.c',
        'src/itty-feed-model.c',
        'src/itty-inference.c',
        'src/itty-manager.c',
        'src/itty-model-metrics.c',
        'src/itty-network.c',
        'src/itty-pipeline.c',
        'src/itty-position.c',
        'src/itty-training-observation.c',
        'src/itty-vocabulary.c',
        'src/itty-work-queue.c',
]

libittybitty = library('ittybitty',
                       sources: library_sources,
                       include_directories: include_directories('src'),
                       dependencies: math_dep)

executable ('itty-bitty',
            sources: 'src/main.c',
            link_with: libittybitty,
            install: true)

test_sources = [
        'src/tests/test-itty-affinity.c',
        'src/tests/test-itty-bit-string.c',
        'src/tests/test-itty-bit-string-list.c',
        'src/tests/test-itty-bit-string-map.c',
        'src/tests/test-itty-exec-buffer.c',
        'src/tests/test-itty-feed-model.c',
        'src/tests/test-itty-inference.c',
        'src/tests/test-itty-manager.c',
        'src/tests/test-itty-model-metrics.c',
        'src/tests/test-itty-network.c',
        'src/tests/test-itty-pipeline.c',
        'src/tests/test-itty-position.c',
        'src/tests/test-itty-training-observation.c',
        'src/tests/test-itty-vocabulary.c',
        'src/tests/test-itty-work-queue.c'
]

foreach test_source : test_sources
    test_name = test_source.split('/')[-1].split('.')[0]
    test_executable = executable(test_name,
                                 sources: [test_source],
                                 include_directories: include_directories('src'),
                                 link_with: libittybitty)
    test_timeout = test_name == 'test-itty-training-observation' or test_name == 'test-itty-feed-model' ? 120 : 30
    test(test_name + '-test', test_executable, timeout: test_timeout)
endforeach
