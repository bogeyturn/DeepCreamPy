from config import get_args

if __name__ == '__main__':
    args = get_args()
    if args.ui_mode:
        import uvicorn
        from server import app
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        raise Exception("Not implemented")