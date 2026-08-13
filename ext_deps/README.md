# `ext_deps/` local install root for Workaholic-Willys external dependencies

> [!NOTE]
> This directory contains the local installation root for all external dependencies required by Workaholic-Willy.
> **ISAAC-SIM is not included** in this directory and must be installed separately.
> Please refer to the [ISAAC-SIM installation instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html) for guidance.
> **Reminder: Watch out if your system is meeting the:**
> See here for help: [ISAAC-SIM system requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).*

> [!WARNING]
> None of the external dependencies are hard ones.
> We still recommend installing **CuRobo** and **Coal** especially if you are planning to use
> this project for real robotics applications.
> We are not responsible for any issues or damages that can occur on your Robot or system.
> **Always be careful and follow the instructions provided.**
> **We highly recommend testing your setup thoroughly in simulation before deploying to real hardware.**


# Layout & Lockfiles

|Subfolder|Description|Wired via|
|---------|-----------|---------|
|`micromamba/`|private package manager| - |
|`locks/`|conda package pinned by URL&hash| cosumed by install script|
|`curobo_env/`|CuRobo conda environment| `WILLY_CUROBO_PYTHON` -> `ext_deps/curobo_env/` -> `python.exe`|
|`curobo/`|CuRobo source code| - |
|`coal_env/`|Coal conda environment| `WILLY_COAL_PYTHON` -> `ext_deps/coal_env/`|

## Install

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_ext_deps.ps1
```

From nothing to a verified stack: it bootstraps micromamba into this folder, builds both environments
from `locks/`, clones and pins cuRobo, installs **both** of its kernel backends, generates the ur5e +
ur3e descriptors, and ends by running the doctor — so its **exit code means "this box can plan"**, not
"the downloads finished". `-Clean` deletes the targets first, which is how you test that it really does
rebuild from nothing. `-Component coal|curobo` does one of them.

```bash
python -m backend.src.robot.safety.planning --doctor   # 0 healthy | 1 degraded | 2 blocked by policy
```


> **The lockfiles are not a nicety.** On Windows, Smart App Control refuses unsigned native code that
> Microsoft's reputation service does not vouch for, and a conda build published days ago usually has no
> reputation yet — so an unpinned `micromamba create` can produce an environment that installs perfectly
> and then cannot load. The pins are chosen to be builds that load. Read
> [`docs/code-integrity.md`](../docs/code-integrity.md) before changing them; re-lock with
> `micromamba env export --explicit` and re-run the doctor.