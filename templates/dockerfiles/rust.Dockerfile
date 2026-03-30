FROM rust:1.78-slim
WORKDIR /app
COPY . .
RUN cargo build --release
CMD ["./target/release/app"]
