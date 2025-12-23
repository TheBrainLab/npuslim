from npuslim.engine.ptq_engine import PTQEngine


def main():
    engine = PTQEngine()
    engine.run()
    engine.save()


if __name__ == "__main__":
    main()