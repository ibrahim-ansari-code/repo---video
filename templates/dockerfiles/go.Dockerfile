FROM golang:1.22-alpine
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN go build -o /app/bin/app .
CMD ["/app/bin/app"]
