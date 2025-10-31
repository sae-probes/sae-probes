# CHANGELOG

## v0.3.0 (2025-10-31)

### Feature

* feat: adding option to customize the mean-diff normalization (#5)

This PR allows the user to further customize the normalization done for
the mean-diff sorting, or disable it entirely. ([`b671860`](https://github.com/sae-probes/sae-probes/commit/b671860d745b1501088cd3a43d279bdf08407f81))

## v0.2.2 (2025-10-30)

### Fix

* fix: handle negative SAE acts when normalizing (#4)

This PR takes the absolute value of activations when calculating their
mean for normalizing mean-diff sorting. This should have no effect for
current SAEs, but should help avoid dividing by 0 for SAEs that can take
on negative activation values, like [AbsTopK
SAEs](https://arxiv.org/abs/2510.00404). ([`aeb2ce7`](https://github.com/sae-probes/sae-probes/commit/aeb2ce7582e5bbe79cb7607e7b5cb8b559d5c25b))

## v0.2.1 (2025-10-04)

### Fix

* fix: load processed and zipped csv rather than raw data (#3)

This PR fixes a bug where we&#39;re trying to load the raw CSV dataset
rather than dataset included in the packaged wheel ([`471e57a`](https://github.com/sae-probes/sae-probes/commit/471e57af8d21096620bdd78c76268a5f30a6d583))

## v0.2.0 (2025-08-27)

### Feature

* feat: adding option to restrict the datasets being run (#2) ([`1c8ba69`](https://github.com/sae-probes/sae-probes/commit/1c8ba69c52a1ee190d25207180350b770d044312))

## v0.1.5 (2025-08-24)

### Fix

* fix: fix formatting for 94_gen_ai csv file (#1) ([`e15b78d`](https://github.com/sae-probes/sae-probes/commit/e15b78dedd6ea9828ebb85abfc2bca34c041c300))

## v0.1.4 (2025-08-21)

### Fix

* fix: removing implicit matplotlib dep from main code ([`7b4d4be`](https://github.com/sae-probes/sae-probes/commit/7b4d4be5d4841b5a149a5555efd18a829a79e69b))

* fix: remove sae-bench dependency ([`ed6833a`](https://github.com/sae-probes/sae-probes/commit/ed6833a74bab65ce440357e019218562e9818add))

## v0.1.3 (2025-08-16)

### Performance

* perf: switch to zstd to compress data files ([`1b14a11`](https://github.com/sae-probes/sae-probes/commit/1b14a119ca6d530720702cd881bfd612a16c566e))

## v0.1.2 (2025-08-16)

### Fix

* fix: adding more metadata to project and README ([`22b8aad`](https://github.com/sae-probes/sae-probes/commit/22b8aad95130e58f7a3640f15c15d53c273fb074))

## v0.1.1 (2025-08-16)

### Fix

* fix: reducing dataset sizes to fit in pypi limit ([`534ffce`](https://github.com/sae-probes/sae-probes/commit/534ffce080d4935537c82d49ad4a720c64295599))

## v0.1.0 (2025-08-16)

### Feature

* feat: publish to PyPI ([`d0a6130`](https://github.com/sae-probes/sae-probes/commit/d0a6130001eebf062474145378a59bf6e24b48cf))

### Unknown

* initial commit ([`110600d`](https://github.com/sae-probes/sae-probes/commit/110600da66033ee7945ce173d71ca098ca0f8db4))

* Initial commit ([`389a610`](https://github.com/sae-probes/sae-probes/commit/389a610dd3f2a0f72eacc1b059c5b824f0c0f558))
