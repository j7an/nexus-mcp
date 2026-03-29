FROM ghcr.io/anomalyco/opencode:latest
ENTRYPOINT ["opencode", "serve"]
CMD ["--hostname", "0.0.0.0", "--port", "4096"]
