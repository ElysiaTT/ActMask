# Milestone 4V plan

4V is a minimal visual counterfactual benchmark and model-headroom audit. It may use only procedural geometry, RGB-D/depth/point-cloud observations, fixed cameras, preregistered corruptions, fair lightweight visual baselines, and frozen execution semantics. It must not modify the completed 3S/3T state-only evidence or train the later specialized model.

Stage A is qualified with the platform EGL ICD configuration. Every visual command must set `VK_ICD_FILENAMES=/etc/vulkan/icd.d/test_nvidia_icd.json` and `XDG_RUNTIME_DIR=/tmp`; the default GLX ICD remains nonfunctional. The three procedural GPU-PhysX task families now expose a fixed 128×128 camera with RGBA, PositionSegmentation-derived depth, and point-cloud modes. Before generating the bounded three-family pilot, freeze the RGB-D serialization, geometry-only feature extractor, storage estimate, split grouping, and hidden-field audit. No state observation may be substituted for an advertised visual observation.
