import json
from typing import Any, Dict, Optional

from botocore.exceptions import ClientError

from mongoagent.cached_query_processor import CachedQueryProcessor


def _is_complete_input(text: str) -> bool:
    """Check if input appears to be complete (simple heuristic)."""
    if not text:
        return True

    complete_indicators = ['.', '?', '!', '"', "'", ')', '}', ']']
    incomplete_indicators = ['(', '{', '[', '"', "'"]

    if text[-1] in complete_indicators:
        return True

    if text[-1] in incomplete_indicators:
        return False

    return True

def handle_message(message, status="Processing") -> None:
    """Handle incoming messages from the server."""
    if isinstance(message, Exception):
        print(f"Error in message handler: {message}")
        return    
    print(message)

def _get_clean_input(prompt: str = "Question: ") -> str:
    """Get clean input from user, handling buffer issues and multi-line text."""
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    if sys.stdin.isatty():
        try:
            import select
            import tty
            import termios

            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                old_settings = termios.tcgetattr(sys.stdin)
                try:
                    tty.setcbreak(sys.stdin.fileno())
                    while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        sys.stdin.read(1)
                finally:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except (ImportError, OSError):
            pass

    try:
        user_input = input(prompt).strip()

        while user_input and not _is_complete_input(user_input):
            print("Input appears incomplete. Continue or press Enter to submit:")
            continuation = input("... ").strip()
            if not continuation:
                break
            user_input += " " + continuation

        return user_input
    except EOFError:
        raise KeyboardInterrupt


def write_dict_to_json_file(data_dict: Dict[str, Any], filename: str) -> None:
    """Write a Python dictionary to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        print(f"Dictionary successfully written to {filename}")
    except Exception as e:
        print(f"Error writing dictionary to JSON file: {e}")


def run_cli(processor: Optional[CachedQueryProcessor] = None) -> None:
    """Run interactive command-line chat loop."""
    proc = processor or CachedQueryProcessor(handle_message)
    proc.set_show_response_progress(False)

    print("Enhanced QueryProcessor with Caching enabled")
    print("Enter questions (Press Ctrl+C to stop):")
    print("Commands:")
    print("  clear - Clear conversation history and caches")
    print("  cache stats - Show cache statistics")
    print("  cache clear - Clear all caches")
    print("  <question> - Claude query with MCP tool support and caching")
    print("Note: For multi-line input, the system will detect incomplete input and prompt for continuation.")

    try:
        answer, _history = proc.query_claude_with_mcp_tools("what collections are available?")
        print(f"first Answer: {answer}")
        answer, _history = proc.query_claude_with_mcp_tools("what claim status for nicole weber?")
        print(f"secondAnswer: {answer}")
        
        while True:
            user_input = _get_clean_input("Question: ")
            answer = "unknown"

            if not user_input:
                answer = "Not a valid question"
            elif user_input.startswith("clear"):
                proc.history = None
                proc.clear_all_caches()
                answer = "History and caches cleared..."
            elif user_input.startswith("cache stats"):
                stats = proc.get_cache_stats()
                answer = f"Cache Statistics: {json.dumps(stats, indent=2)}"
            elif user_input.startswith("cache clear"):
                proc.clear_all_caches()
                answer = "All caches cleared"
            else:
                answer, _history = proc.query_claude_with_mcp_tools(user_input)
                answer = None

            if answer:
                print(f"Answer: {answer}")

    except ClientError as error:
        error_code = error.response['Error']['Code']
        if error_code in ['ExpiredTokenException', 'ExpiredToken']:
            proc.message_handler(f"AWS Token has expired! {error}", error)
        elif error_code == 'ValidationException':
            proc.history = None
            proc.clear_all_caches()
            proc.message_handler(f"Too much history, clearing... {error}", error)
            run_cli(proc)
        else:
            proc.message_handler(f"Some other AWS client error occurred: {error.response}", error)
    except KeyboardInterrupt:
        proc.message_handler("\nKeyboard interrupt received, exiting...")
        proc.message_handler(f"Final cache stats: {proc.get_cache_stats()}")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
