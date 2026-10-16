# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is a copy of `eqasim-java` (the Java part of the eqasim project), used here for an ABMT 2026 course assignment. eqasim extends the agent-based transport simulation framework [MATSim](https://matsim.org/) with ready-to-run scenarios and a discrete mode choice layer. Scenario input data (populations, networks, schedules, configs) is not in this repo; it comes from the separate pipeline repositories and is passed in via `--config-path`.

## Build & test

Maven multi-module project, Java 25 (set in the root `pom.xml`), MATSim version pinned by the `matsim.version` property.

```bash
mvn test -B -Dmatsim.preferLocalDtds=true          # what CI runs (all modules)
mvn package -DskipTests                            # build all modules
mvn -pl core test -Dtest=TestVolumeDelayFunction   # one test class in one module
mvn -pl core test -Dtest=TestSimulationPipeline#testDrt   # a single test method
mvn -pl switzerland -am package -Pstandalone -DskipTests       # fat jar via the "standalone" (shade) profile
```

- `-am` is needed when building a downstream module (e.g. `switzerland`) so that `core` is built too.
- Tests use JUnit 4. Most tests live in `core`. `TestSimulationPipeline` and similar tests copy the small `melun` fixture from `core/src/test/resources/melun` into a `melun_test/` working directory and run short simulations, so they are slow.
- There is no linter or formatter configured.

## Modules

- `core`: all reusable eqasim functionality (simulation setup, mode choice framework, VDF, DRT/feeder, emissions, policies, termination, analysis, scenario cutting, export tools).
- `switzerland`, `ile_de_france`, `sao_paulo`, `san_francisco`, `los_angeles`: scenario-specific modules that depend on `core` and add their own configurator, mode choice models and `RunSimulation` entry points. `switzerland` contains both a nationwide model (`ch`) and the Zurich model (`zurich`), which reuses and extends the Swiss one.
- `server`: a separate module, also built on `core`.
- `gis/`: shapefiles (Zurich, Paris, ...) used as inputs, e.g. for cutting scenarios. It is not a Maven module.

## Architecture

### Configurator pattern
Every simulation is wired through a subclass of `core/.../simulation/EqasimConfigurator.java`. Its constructor registers config groups, Guice `AbstractModule`s, QSim modules and QSim component configurators. Many of them are *optional* and only activate when their config group is present in the config XML (e.g. DRT, feeder DRT, VDF, abstract access, termination, leg time constraint, emissions). Scenario configurators (e.g. `ZurichConfigurator`, `SwitzerlandConfigurator`) call `super(cmd)` and then register their own mode choice modules.

A `RunSimulation` entry point follows this sequence, and the order matters:
1. create the configurator and `ConfigUtils.loadConfig(...)` using its config groups
2. `configurator.configure(config)` (scenario-specific config tweaks), then `cmd.applyConfiguration(config)` (CLI `--config:...` overrides)
3. `ScenarioUtils.createScenario`, then `configurator.configureScenario`, then `ScenarioUtils.loadScenario`, then `configurator.adjustScenario` (e.g. copying household attributes onto persons)
4. `new Controler(scenario)`, then `configurator.configureController(controller)`, then `controller.run()`

Generic tools find a configurator through `EqasimConfigurator.getInstance(cmd)`: the `--eqasim-configurator` option, the `eqasim_configurator` environment variable, or `configurator=` in an `eqasim.properties` resource on the classpath. Note that `switzerland/src/main/resources/eqasim.properties` points to `org.eqasim.switzerland.SwitzerlandConfigurator`, but the class actually lives in `org.eqasim.switzerland.ch`.

### Mode choice
eqasim builds on the MATSim `discrete_mode_choice` contrib. Scenario modules extend `AbstractEqasimExtension` and register named components with `bindUtilityEstimator(NAME)`, `bindCostModel(NAME)`, `bindTripConstraintFactory(NAME)` and `bindModeAvailability(NAME)`. The configurator then selects these components by name in `EqasimConfigGroup` (`setEstimator(mode, NAME)`, cost models) and `DiscreteModeChoiceConfigGroup` (constraints, mode availability).

Utility models are split into three layers, which you can see under `.../mode_choice/utilities/` in both `core` and each scenario:
- **variables**: plain data holders for the attributes of a trip, person or mode
- **predictors**: compute the variables from a trip/person (they are usually cached)
- **estimators**: combine the variables with **parameters** into a utility. Scenario estimators typically extend the core/Swiss estimators and add terms.

Mode parameters are classes such as `ZurichModeParameters` with hard-coded defaults (`buildFrom...()`). Their values can be overridden from a YAML file (`eqasim:modeParametersPath`) and from the command line (`--mode-parameter:car.betaTravelTime_u_min=...`) through `ParameterDefinition`. The same mechanism applies to `--cost-parameter:...`.

### Other core subsystems (see `docs/`)
- `scenario/cutter`: cuts a smaller scenario out of a larger one (`docs/cutting.md`)
- `simulation/modes/drt`, `feeder_drt`: on-demand mobility (`docs/on_demand_mobility.md`)
- `standalone_mode_choice`: runs mode choice without a full simulation (`docs/standalone_mode_choice.md`)
- `simulation/vdf`: volume-delay-function network simulation as an alternative to the QSim car simulation (`docs/vdf.md`)

## Conventions

- PR titles must follow Conventional Commits (checked in CI); commit history uses e.g. `chore(core): ...`, `feat(...)`, `fix(...)`.
- Versions are managed by release-please. Do not edit the version strings marked `x-release-please-version` / `x-release-please-start-version` by hand, and leave `CHANGELOG.md` to release-please.