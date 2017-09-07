# load the environment file that contains the WIT_URL
source .env

wget -O wit_data.zip $WIT_URL
rm -rf BotCycle
unzip -o wit_data.zip
rm -rf wit_data.zip
