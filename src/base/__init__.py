import argparse
import asyncio
import inspect
import logging
import os
from typing import Callable
from fastmcp import FastMCP
from fastmcp.server.server import Transport
from fastmcp.server.middleware.logging import LoggingMiddleware

import tracemalloc

tracemalloc.start()

mcp: FastMCP = FastMCP()  # Decorators can import this
mcp.add_middleware(LoggingMiddleware(logging.getLogger("MIDDLEWARE"), log_level=logging.INFO, include_payloads=True, ))

class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=True, default=None, **kwargs):
        if envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required, 
                                         **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

def is_safe_async_callable(var):
    if not callable(var):
        return False
    if inspect.iscoroutinefunction(var):
        return True
    try:
        result = var()
        return inspect.isawaitable(result)
    except TypeError:
        # Probably requires arguments; can’t confirm awaitability
        return False

def parse_args(server_name: str, server_type: str, add_args_fn=None, add_mcp_args: bool = True):
    logger = logging.getLogger("parse_args")
    logger.info("Parsing args")
    parser = argparse.ArgumentParser(description=f"Start the {server_name} {server_type}.")
    if add_mcp_args:
        parser.add_argument(
            "-t", "--transport",
            choices=["stdio", "sse", "streamable-http"],
            default="stdio",
            help="Transport mechanism to use for FastMCP",
            action=EnvDefault,
            envvar='MCP_TRANSPORT'
        )
        parser.add_argument(
            "-b", "--bind",
            type=str,
            default=None,
            help="IP address to bind the server to (required if transport is not stdio)",
            action=EnvDefault,
            envvar='MCP_BIND'
        )
        parser.add_argument(
            "-p", "--port",
            type=int,
            default=None,
            help="Port number to bind the server to (required if transport is not stdio)",
            action=EnvDefault,
            envvar='MCP_PORT'
        )
        parser.add_argument(
            "-m", "--mount",
            type=str,
            default=None,
            help="Path to mount the server on (only for streamable-http transport)",
            action=EnvDefault,
            envvar='MCP_MOUNT'
        )
    parser.add_argument(
        "-l", "--log-level",
        type=str,
        default="info",
        help="Logging level for the server (e.g., debug, info, warning, error, critical)",
        action=EnvDefault,
        envvar='LOG_LEVEL'
    )

    # Allow the child to inject more args
    if add_args_fn:
        parser = add_args_fn(parser)

    args = parser.parse_args()
    
    logger.info('Parsed arguments: %s', args)

    if add_mcp_args:
        if args.transport != "stdio":
            if not args.bind:
                parser.error("--bind is required when transport is not 'stdio'")
            if not args.port:
                parser.error("--port is required when transport is not 'stdio'")

    return args

def run_app(service_name: str, service_type: str, workload_fn: Callable, is_mcp: bool = False, add_args_fn: Callable | None =None, run_callback: Callable | None =None):
    args = parse_args(service_name, service_type, add_args_fn, is_mcp)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), args.log_level))
    logger = logging.getLogger("run_app")
    async def run():
        # Let the child do any last-minute config
        if run_callback:
            logger.info("Running 'run_callback'...")
            if is_safe_async_callable(run_callback):
                await run_callback(args)
            else:
                run_callback(args)

        logger.info('Running workload...')
        if is_safe_async_callable(workload_fn):
            await workload_fn(**args.__dict__)
        else:
            workload_fn(**args.__dict__)
    asyncio.run(run())

def run_server(server_name: str, add_args_fn: Callable | None=None, run_callback: Callable | None=None):
    async def run_mcp(transport: Transport, bind: str, port: str, mount: str, log_level: str, **kwargs):
        # Implement MCP server logic here
        await mcp.run_async(transport=transport, host=bind, port=port, path=mount, log_level=log_level)

    run_app(service_name=server_name, service_type="Fast MCP Server", workload_fn=run_mcp, is_mcp=True, add_args_fn=add_args_fn, run_callback=run_callback)

def run_daemon(server_name: str, workload_fn: Callable, add_args_fn: Callable | None = None, run_callback: Callable | None = None):
    run_app(service_name=server_name, service_type="Daemon", workload_fn=workload_fn, add_args_fn=add_args_fn, run_callback=run_callback)
