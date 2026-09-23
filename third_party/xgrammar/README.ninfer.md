# XGrammar native core

Vendored from https://github.com/mlc-ai/xgrammar, release v0.2.5,
commit `2ea71da4ccb997a06928c9fb69b99f330da56697` (Apache-2.0).
Includes the unchanged native core and public headers. DLPack headers are pinned by that
release to `bbd2f4d32427e548797929af08cfe2a9cbb3cf12`; picojson carries its license in its
header. Upstream licenses are retained. `CMakeLists.txt` is the NInfer build integration.
No Python runtime, model execution or network access is involved at build/run time.
