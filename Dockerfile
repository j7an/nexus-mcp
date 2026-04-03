FROM ghcr.io/anomalyco/opencode:1.3.13
ENTRYPOINT ["opencode", "serve"]
CMD ["--hostname", "0.0.0.0", "--port", "4096"]
