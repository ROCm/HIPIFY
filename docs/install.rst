.. meta::
   :description: Installation instructions for HIPIFY
   :keywords: hipify, hip, cuda, source, code, c++, cpp, rocm, translate, translator, transpile, compile, clang, cuda2hip

.. _installation:

**************
Install HIPIFY
**************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./building/build-hipify-clang-linux` or :doc:`./building/build-hipify-clang-windows`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

HIPIFY is included with the ROCm Core SDK on Linux and Windows. For the most
complete installation, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install the HIPIFY package on Linux
===================================

Alternatively, if you want to install HIPIFY without additional ROCm libraries
and tools, install the ``amdrocm-hipify`` package.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the HIPIFY package that matches your desired ROCm version.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-hipify<rocm_version>

   ``<rocm_version>`` represents the ROCm Core SDK version to install. Omit
   this suffix to install the latest available version.

   For example:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-hipify

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-hipify

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-hipify

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including ROCm Systems
Profiler. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
