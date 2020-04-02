import click


@click.command()
@click.option('--file', '-f', prompt = 'Path to .csv',
              help = 'Path to .csv file from which to load data')
@click.option('--column', '-c', default='message_body',
              help = 'Name of column that contains message body')
def main(file, column):
    click.echo(f'{file} {column}')

if __name__ == '__main__':
    main()
