<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Debug

This file is only the entry point. The Qwen3 megakernel notes are organized by
how they are used during development:

- [Debug map](qwen3-megakernel/index.md): current checkpoints and active
  failure boundary.
- [Symptom lookup](qwen3-megakernel/symptoms.md): start here when a run fails.
- [Diagnostic methods](qwen3-megakernel/diagnostic-methods.md): commands and
  checks that were actually used in this repository.
- [Lessons](qwen3-megakernel/lessons.md): design constraints and preflight
  checks inferred from diagnosed failures.

The rule for this directory is strict: do not record a generic debugging idea
until it has been used to diagnose a real failure in the Qwen3 bring-up.
