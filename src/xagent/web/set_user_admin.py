"""
将指定用户设置为管理员（is_admin=True），拥有所有权限。
用法：
  python -m xagent.web.set_user_admin --username frae
  python -m xagent.web.set_user_admin --username frae --yes
"""
import argparse
import logging
import sys
from typing import cast

from dotenv import load_dotenv

from .models.database import get_db, init_db
from .models.user import User

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将指定用户设置为管理员（is_admin=True）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m xagent.web.set_user_admin --username frae
  python -m xagent.web.set_user_admin --username frae --yes
        """,
    )
    _ = parser.add_argument(
        "--username",
        required=True,
        help="要设为管理员的用户名（如 frae）",
    )
    _ = parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过确认提示",
    )
    return parser.parse_args()


def set_user_admin(username: str, skip_confirmation: bool = False) -> None:
    username = username.strip()
    if not username:
        raise ValueError("用户名不能为空")

    db = next(get_db())
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            raise ValueError(f"用户 '{username}' 不存在，请先注册该账号")

        if getattr(user, "is_admin", False):
            raise ValueError(f"用户 '{username}' 已经是管理员，无需重复设置")

        if not skip_confirmation:
            confirmed = input(
                f"确认将用户 '{username}' 设置为管理员（拥有所有权限）？[y/N]: "
            ).strip()
            if confirmed.lower() not in {"y", "yes"}:
                raise ValueError("已取消操作")

        setattr(user, "is_admin", True)
        db.commit()
    except ValueError:
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    username = cast(str, args.username).strip()
    yes_arg = cast(bool, args.yes)

    try:
        init_db()
        set_user_admin(username, skip_confirmation=yes_arg)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error("设置管理员失败: %s", exc)
        sys.exit(1)

    print(f"已将用户 '{username}' 设置为管理员，拥有所有权限。")


if __name__ == "__main__":
    main()
