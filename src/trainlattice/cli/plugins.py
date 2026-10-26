"""Presentation of metadata-only plugin diagnostics."""

import argparse
import json
from pathlib import Path

from trainlattice.kernel.bootstrap import BootstrapConfig, read_yaml
from trainlattice.kernel.lifecycle import PluginManager
from trainlattice.kernel.manifest import PluginManifest
from trainlattice.kernel.snapshot import PluginSnapshot
from trainlattice.kernel.sources import PluginSourceResolver


def add_plugin_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add static plugin inspection commands to the bootstrap CLI."""
    parser = subparsers.add_parser("plugins", help="inspect plugin metadata without activation")
    commands = parser.add_subparsers(dest="plugin_command", required=True)
    inspect = commands.add_parser("inspect", help="inspect a plugin manifest")
    inspect.add_argument("source")
    validate = commands.add_parser("validate", help="validate manifests and their dependency graph")
    validate.add_argument("sources", nargs="*")
    validate.add_argument("--config", type=Path)
    for command in ("list", "graph"):
        child = commands.add_parser(command, help=f"show the effective plugin {command}")
        child.add_argument("--config", type=Path, required=True)


def dispatch_plugins(args: argparse.Namespace) -> int:
    """Display declarative plugin metadata without importing entry files."""
    if args.plugin_command == "inspect":
        source = PluginSourceResolver().resolve(args.source)
        manifest = PluginManifest.read(source.root)
        print(json.dumps({"id": manifest.id, "version": manifest.version, "entry": str(manifest.entry_file.relative_to(manifest.root)) + ":" + manifest.entry_callable, "provides": [item.service for item in manifest.provides], "requires": {alias: item.service for alias, item in manifest.requires.items()}, "state": manifest.state_mode}, indent=2, ensure_ascii=False))
        return 0
    config_path = getattr(args, "config", None)
    if config_path is not None:
        if getattr(args, "sources", []):
            raise ValueError("plugins validate accepts sources or --config, not both")
        bootstrap = BootstrapConfig.from_mapping(read_yaml(config_path))
        plan = PluginManager(PluginSourceResolver(config_path.resolve().parent)).plan(bootstrap.plugins, bindings=bootstrap.bindings)
    else:
        if not args.sources:
            raise ValueError("plugins validate requires source directories or --config")
        plan = PluginManager().plan(args.sources)
    if args.plugin_command == "validate":
        print(f"Valid plugin graph: {len(plan.plugins)} plugins")
    elif args.plugin_command == "list":
        print(json.dumps(PluginSnapshot.from_plan(plan).to_dict(), indent=2, ensure_ascii=False))
    else:
        providers = {item.service: plugin.manifest.id for plugin in plan.plugins for item in plugin.manifest.provides}
        for plugin in plan.plugins:
            print(plugin.manifest.id)
            for alias, service in plugin.bindings.items():
                if service is not None:
                    print(f"  {providers[service]} -> {plugin.manifest.id} ({alias}: {service})")
    return 0
