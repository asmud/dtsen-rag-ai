#!/bin/bash

# DTSEN RAG AI - Database Setup Script
# This script sets up the PostgreSQL database with pgvector for the DTSEN RAG AI system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-rag_db}"
DB_USER="${DB_USER:-rag_user}"
DB_PASSWORD="${DB_PASSWORD:-rag_pass}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-dtsen_rag_postgres}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if PostgreSQL is running
check_postgres() {
    if docker ps | grep -q "$DOCKER_CONTAINER"; then
        print_success "PostgreSQL container is running"
        return 0
    elif command_exists psql; then
        if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c '\q' 2>/dev/null; then
            print_success "PostgreSQL is accessible"
            return 0
        fi
    fi
    return 1
}

# Function to execute SQL file
execute_sql() {
    local sql_file="$1"
    local description="$2"
    
    print_info "$description"
    
    if [ ! -f "$sql_file" ]; then
        print_error "SQL file not found: $sql_file"
        return 1
    fi
    
    # Try Docker first, then direct psql
    if docker ps | grep -q "$DOCKER_CONTAINER"; then
        if docker exec -i "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" < "$sql_file"; then
            print_success "$description completed"
            return 0
        fi
    elif command_exists psql; then
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$sql_file"; then
            print_success "$description completed"
            return 0
        fi
    fi
    
    print_error "$description failed"
    return 1
}

# Function to run Python verification
run_verification() {
    local connection_string="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"
    
    print_info "Running database verification..."
    
    if command_exists python3; then
        if python3 "$SCRIPT_DIR/verify_setup.py" --connection-string "$connection_string"; then
            print_success "Database verification passed"
            return 0
        else
            local exit_code=$?
            if [ $exit_code -eq 2 ]; then
                print_warning "Database verification passed with warnings"
                return 0
            else
                print_error "Database verification failed"
                return 1
            fi
        fi
    else
        print_warning "Python3 not available, skipping verification"
        return 0
    fi
}

# Function to display usage
usage() {
    cat << EOF
DTSEN RAG AI Database Setup Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    init         Initialize database schema (default)
    test         Load test data
    verify       Verify database setup
    reset        Reset database (drops and recreates schema)
    full         Run init + test + verify

Options:
    -h, --help          Show this help message
    --db-host HOST      Database host (default: localhost)
    --db-port PORT      Database port (default: 5432)
    --db-name NAME      Database name (default: rag_db)
    --db-user USER      Database user (default: rag_user)
    --db-password PASS  Database password (default: rag_pass)
    --docker-container  Docker container name (default: dtsen_rag_postgres)
    --skip-verification Skip verification step
    --force             Skip confirmation prompts

Environment variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DOCKER_CONTAINER

Examples:
    $0                                          # Initialize with defaults
    $0 init                                     # Initialize schema only
    $0 test                                     # Load test data
    $0 verify                                   # Verify setup
    $0 full                                     # Complete setup and test
    $0 --db-host myhost.com --db-port 5433 init
EOF
}

# Parse command line arguments
COMMAND="init"
SKIP_VERIFICATION=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        --db-host)
            DB_HOST="$2"
            shift 2
            ;;
        --db-port)
            DB_PORT="$2"
            shift 2
            ;;
        --db-name)
            DB_NAME="$2"
            shift 2
            ;;
        --db-user)
            DB_USER="$2"
            shift 2
            ;;
        --db-password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        --docker-container)
            DOCKER_CONTAINER="$2"
            shift 2
            ;;
        --skip-verification)
            SKIP_VERIFICATION=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        init|test|verify|reset|full)
            COMMAND="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
print_info "DTSEN RAG AI Database Setup"
print_info "=========================="
print_info "Database: $DB_HOST:$DB_PORT/$DB_NAME"
print_info "User: $DB_USER"
print_info "Command: $COMMAND"
echo

# Check prerequisites
if ! check_postgres; then
    print_error "PostgreSQL is not accessible. Please ensure:"
    echo "  1. PostgreSQL is running (docker-compose up -d postgres)"
    echo "  2. Connection parameters are correct"
    echo "  3. Database exists and user has permissions"
    exit 1
fi

# Confirmation prompt (unless forced)
if [ "$FORCE" != true ]; then
    echo -n "Continue with database setup? [y/N] "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled"
        exit 0
    fi
fi

# Execute commands
case $COMMAND in
    init)
        execute_sql "$SCRIPT_DIR/init.sql" "Initializing database schema"
        if [ "$SKIP_VERIFICATION" != true ]; then
            run_verification
        fi
        ;;
    
    test)
        execute_sql "$SCRIPT_DIR/test_setup.sql" "Loading test data"
        ;;
    
    verify)
        run_verification
        ;;
    
    reset)
        print_warning "This will drop and recreate the database schema!"
        if [ "$FORCE" != true ]; then
            echo -n "Are you sure? [y/N] "
            read -r response
            if [[ ! "$response" =~ ^[Yy]$ ]]; then
                print_info "Reset cancelled"
                exit 0
            fi
        fi
        
        # Drop and recreate tables
        print_info "Resetting database schema..."
        
        reset_sql=$(cat << 'EOF'
-- Drop existing objects
DROP TABLE IF EXISTS data_rag_kb CASCADE;
DROP TABLE IF EXISTS rag_system_info CASCADE;
DROP VIEW IF EXISTS vector_store_stats CASCADE;
DROP FUNCTION IF EXISTS check_vector_store_health() CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
EOF
)
        
        echo "$reset_sql" | if docker ps | grep -q "$DOCKER_CONTAINER"; then
            docker exec -i "$DOCKER_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME"
        else
            PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
        fi
        
        execute_sql "$SCRIPT_DIR/init.sql" "Reinitializing database schema"
        
        if [ "$SKIP_VERIFICATION" != true ]; then
            run_verification
        fi
        ;;
    
    full)
        execute_sql "$SCRIPT_DIR/init.sql" "Initializing database schema"
        execute_sql "$SCRIPT_DIR/test_setup.sql" "Loading test data"
        run_verification
        ;;
    
    *)
        print_error "Unknown command: $COMMAND"
        usage
        exit 1
        ;;
esac

print_success "Database setup completed successfully!"

# Display next steps
echo
print_info "Next steps:"
echo "  1. Start the DTSEN RAG AI application:"
echo "     docker-compose up -d"
echo
echo "  2. Check application health:"
echo "     curl http://localhost:8000/health"
echo
echo "  3. View API documentation:"
echo "     open http://localhost:8000/docs"
echo
echo "  4. Test the chat endpoint:"
echo "     curl -X POST http://localhost:8000/chat \\"
echo "          -H 'Content-Type: application/json' \\"
echo "          -d '{\"question\": \"Hello, how are you?\"}'"
echo