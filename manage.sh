#!/bin/bash

COMPOSE_FILE="docker-compose.production.yml"

case "$1" in
    start)
        echo "🚀 Starting DGDN production services..."
        docker-compose -f $COMPOSE_FILE up -d
        ;;
    stop)
        echo "⏹️ Stopping DGDN production services..."
        docker-compose -f $COMPOSE_FILE down
        ;;
    restart)
        echo "🔄 Restarting DGDN production services..."
        docker-compose -f $COMPOSE_FILE restart
        ;;
    status)
        echo "📊 DGDN production services status:"
        docker-compose -f $COMPOSE_FILE ps
        ;;
    logs)
        echo "📝 DGDN production logs:"
        docker-compose -f $COMPOSE_FILE logs -f "${2:-dgdn-api}"
        ;;
    update)
        echo "🔄 Updating DGDN production deployment..."
        docker-compose -f $COMPOSE_FILE pull
        docker-compose -f $COMPOSE_FILE up -d
        ;;
    backup)
        echo "💾 Creating backup..."
        mkdir -p backups/$(date +%Y%m%d_%H%M%S)
        cp -r data models logs backups/$(date +%Y%m%d_%H%M%S)/
        echo "✅ Backup created in backups/"
        ;;
    health)
        echo "🔍 Health check:"
        curl -s http://localhost/health | jq '.' || echo "❌ Health check failed"
        ;;
    metrics)
        echo "📊 Current metrics:"
        curl -s http://localhost/metrics | grep dgdn || echo "❌ Metrics unavailable"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|update|backup|health|metrics}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  status  - Show service status"
        echo "  logs    - Show logs (specify service name as 2nd arg)"
        echo "  update  - Update and restart services"
        echo "  backup  - Create backup of data/models/logs"
        echo "  health  - Check API health"
        echo "  metrics - Show current metrics"
        exit 1
        ;;
esac
