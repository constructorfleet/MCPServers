import argparse
import asyncio
from inspect import isawaitable
import inspect
import logging
from fastmcp import FastMCP
from fastmcp.server.middleware.logging import LoggingMiddleware

mcp: FastMCP = FastMCP()  # Decorators can import this
mcp.add_middleware(LoggingMiddleware(logging.getLogger("MIDDLEWARE"), log_level=logging.INFO, include_payloads=True, ))

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

def parse_args(server_name: str, add_args_fn=None):
    parser = argparse.ArgumentParser(description=f"Start the {server_name} FastMCP server.")

    parser.add_argument(
        "-t", "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport mechanism to use for FastMCP",
    )
    parser.add_argument(
        "-b", "--bind",
        type=str,
        default=None,
        help="IP address to bind the server to (required if transport is not stdio)",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=None,
        help="Port number to bind the server to (required if transport is not stdio)",
    )
    parser.add_argument(
        "-m", "--mount",
        type=str,
        default=None,
        help="Path to mount the server on (only for streamable-http transport)",
    )
    parser.add_argument(
        "-l", "--log-level",
        type=str,
        default="info",
        help="Logging level for the server (e.g., debug, info, warning, error, critical)",
    )

    # Allow the child to inject more args
    if add_args_fn:
        parser = add_args_fn(parser)

    args = parser.parse_args()

    if args.transport != "stdio":
        if not args.bind:
            parser.error("--bind is required when transport is not 'stdio'")
        if not args.port:
            parser.error("--port is required when transport is not 'stdio'")

    return args

def run_server(server_name: str, add_args_fn=None, run_callback=None):
    args = parse_args(server_name, add_args_fn)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), "INFO"))

    async def run():
        # Let the child do any last-minute config
        if run_callback:
            if is_safe_async_callable(run_callback):
                await run_callback(args)
            else:
                run_callback(args)

        if args.transport == "stdio":
            await mcp.run_async(transport="stdio")
        elif args.transport == "sse":
            await mcp.run_async(
                transport="sse",
                host=args.bind,
                port=args.port,
                path=args.mount,
                log_level=args.log_level,
            )
        elif args.transport == "streamable-http":
            await mcp.run_async(
                transport="http",
                host=args.bind,
                port=args.port,
                path=args.mount,
                log_level=args.log_level,
            )
        else:
            raise ValueError(f"Unsupported transport: {args.transport}")
    asyncio.run(run())
