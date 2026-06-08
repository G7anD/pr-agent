# Puppeteer's official image ships a matching Chrome + sets PUPPETEER_EXECUTABLE_PATH.
FROM ghcr.io/puppeteer/puppeteer:24

WORKDIR /home/pptruser/app

# Install deps first (layer cache). The base image's cached Chrome matches puppeteer 24.
COPY --chown=pptruser:pptruser package.json ./
RUN npm install --omit=dev

COPY --chown=pptruser:pptruser banner-template.js generate-banner.js server.js ./

ENV PORT=41928
EXPOSE 41928
CMD ["node", "server.js"]
