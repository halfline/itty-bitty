project ('itty-bitty', 'c', default_options : ['c_std=c11'])
add_project_arguments('-D_GNU_SOURCE', language: 'c')
cc = meson.get_compiler('c')
math_dep = cc.find_library('m', required: false)
experimental_prototypes = get_option('experimental_prototypes')
runtime_include_dirs = include_directories('src')
prototype_include_dirs = include_directories(
        'experiments/route-key-selector',
        'experiments/route-key-gray-payload',
        'experiments/self-describing-gray-payload',
)
project_include_dirs = [runtime_include_dirs]

if experimental_prototypes
    project_include_dirs += [prototype_include_dirs]
endif

library_sources = [
        'src/itty-affinity.c',
        'src/itty-bit-string.c',
        'src/itty-bit-string-list.c',
        'src/itty-bit-string-map.c',
        'src/itty-decoder.c',
        'src/itty-exec-buffer.c',
        'src/itty-inference.c',
        'src/itty-manager.c',
        'src/itty-model-metrics.c',
        'src/itty-network.c',
        'src/itty-pipeline.c',
        'src/itty-position.c',
        'src/itty-trainer.c',
        'src/itty-vocabulary.c',
        'src/itty-work-queue.c',
]

if experimental_prototypes
    library_sources += [
        'experiments/route-key-selector/itty-route-key-selector.c',
        'experiments/route-key-gray-payload/itty-route-key-gray-payload.c',
        'experiments/self-describing-gray-payload/itty-self-describing-gray-payload.c',
    ]
endif

libittybitty = library('ittybitty',
                       sources: library_sources,
                       include_directories: project_include_dirs,
                       dependencies: math_dep)

executable ('itty-bitty',
            sources: 'src/decoder/main.c',
            include_directories: project_include_dirs,
            link_with: libittybitty,
            install: true)

executable ('itty-bitty-trainer',
            sources: 'src/trainer/main.c',
            include_directories: project_include_dirs,
            link_with: libittybitty,
            install: true)

executable ('itty-bitty-harness',
            sources: 'src/harness/main.c',
            include_directories: project_include_dirs,
            link_with: libittybitty,
            install: true)

test_sources = [
        'src/tests/test-itty-affinity.c',
        'src/tests/test-itty-bit-string.c',
        'src/tests/test-itty-bit-string-list.c',
        'src/tests/test-itty-bit-string-map.c',
        'src/tests/test-itty-decoder.c',
        'src/tests/test-itty-exec-buffer.c',
        'src/tests/test-itty-inference.c',
        'src/tests/test-itty-manager.c',
        'src/tests/test-itty-model-metrics.c',
        'src/tests/test-itty-network.c',
        'src/tests/test-itty-pipeline.c',
        'src/tests/test-itty-position.c',
        'src/tests/test-itty-vocabulary.c',
        'src/tests/test-itty-work-queue.c'
]

if experimental_prototypes
    test_sources += [
        'experiments/route-key-selector/test-itty-route-key-selector.c',
        'experiments/route-key-gray-payload/test-itty-route-key-gray-payload.c',
        'experiments/self-describing-gray-payload/test-itty-self-describing-gray-payload.c',
    ]
endif

foreach test_source : test_sources
    test_name = test_source.split('/')[-1].split('.')[0]
    test_executable = executable(test_name,
                                 sources: [test_source],
                                 include_directories: project_include_dirs,
                                 link_with: libittybitty)
    test_timeout = 30
    test(test_name + '-test', test_executable, timeout: test_timeout)
endforeach
