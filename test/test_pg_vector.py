import logging
import pytest
import asyncio
from time import sleep
from datetime import datetime
import dotenv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.pg_vector.pg_vector import PGVector
from app.utils.pg_vector.table import OCRTable

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()


class TestPGVector:
    pg_vector = None
    
    def sample_ocr_record(self):
        """Create a sample OCR record for testing"""
        record = OCRTable()
        record.id = "test_id_123"
        record.url = "https://example.com/test.pdf"
        record.markdownUrl = "https://example.com/test.md"
        record.jsonUrl = "https://example.com/test.json"
        record.status = "pending"
        record.createdAt = datetime.now()
        record.updatedAt = datetime.now()
        record.createdBy = "test_user"
        record.updatedBy = "test_user"
        return record
    
    def get_connection_string(self):
        """PostgreSQL connection string"""
        connection_string = os.getenv('POSTGRES_URL_NO_SSL_DEV')
        if connection_string:
            # Convert to async driver format
            if connection_string.startswith('postgresql://'):
                connection_string = connection_string.replace('postgresql://', 'postgresql+asyncpg://', 1)
            return connection_string
        raise ValueError("No connection string found in environment")
    
    def setup_pg_vector(self):
        """Initialize PGVector instance"""
        if not self.pg_vector:
            connection_string = self.get_connection_string()
            self.pg_vector = PGVector(connection_string)
            
    async def close_pg_vector(self):
        """Close PGVector session"""
        if self.pg_vector:
            await self.pg_vector.close()
            self.pg_vector = None
    
    @pytest.mark.asyncio
    async def test_init(self):
        """Test PGVector initialization"""
        self.setup_pg_vector()
        connection_string = self.get_connection_string()
        pg_vector = self.pg_vector

        assert pg_vector.connection_string == connection_string
        assert pg_vector.table == OCRTable
    
    @pytest.mark.asyncio
    async def test_insert_record(self):
        """Test inserting a new record"""
        self.setup_pg_vector()
        pg_vector = self.pg_vector

        try:
            # Ensure table exists
            await pg_vector.ensure_table_exists()
            
            # Create and insert a test record
            record = self.sample_ocr_record()
            result = await pg_vector.upsert_record(record)
            
            assert result is True
            
            # Verify the record was inserted by retrieving it
            retrieved_record = await pg_vector.get_record_by_id(record.id)
            assert retrieved_record is not None
            assert retrieved_record.id == record.id
            assert retrieved_record.url == record.url
            assert retrieved_record.status == record.status
            
        finally:
            # Clean up
            await pg_vector.delete_record("test_id_123")
            # await pg_vector.close()
    
    @pytest.mark.asyncio
    async def test_update_record(self):
        """Test updating an existing record"""
        self.setup_pg_vector()
        pg_vector = self.pg_vector

        try:
            # Ensure table exists
            await pg_vector.ensure_table_exists()
            
            # Insert a test record first
            record = self.sample_ocr_record()
            await pg_vector.upsert_record(record)
            
            # Update the record
            updates = {
                "status": "completed",
                "markdownUrl": "https://example.com/updated_test.md",
                "updatedAt": datetime.now(),
                "updatedBy": "test_user_updated"
            }
            result = await pg_vector.update_record(record.id, updates)
            
            assert result is True
            
            # Verify the record was updated
            updated_record = await pg_vector.get_record_by_id(record.id)
            assert updated_record is not None
            assert updated_record.status == "completed"
            assert updated_record.markdownUrl == "https://example.com/updated_test.md"
            assert updated_record.updatedBy == "test_user_updated"
            
        finally:
            # Clean up
            await pg_vector.delete_record("test_id_123")
            # await pg_vector.close()
    
    @pytest.mark.asyncio
    async def test_delete_record(self):
        """Test deleting a record"""
        self.setup_pg_vector()
        pg_vector = self.pg_vector

        try:
            # Ensure table exists
            await pg_vector.ensure_table_exists()
            
            # Insert a test record first
            record = self.sample_ocr_record()
            await pg_vector.upsert_record(record)
            
            # Verify the record exists
            existing_record = await pg_vector.get_record_by_id(record.id)
            assert existing_record is not None
            
            # Delete the record
            result = await pg_vector.delete_record(record.id)
            assert result is True
            
            # Verify the record was deleted
            deleted_record = await pg_vector.get_record_by_id(record.id)
            assert deleted_record is None
            
        finally:
            # Ensure cleanup (in case test fails)
            await pg_vector.delete_record("test_id_123")
            # await pg_vector.close()
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_record(self):
        """Test retrieving a record that doesn't exist"""
        self.setup_pg_vector()
        pg_vector = self.pg_vector

        try:
            # Ensure table exists
            await pg_vector.ensure_table_exists()
            
            # Try to get a record that doesn't exist
            record = await pg_vector.get_record_by_id("nonexistent_id")
            assert record is None
            
        finally:
            # await pg_vector.close()
            pass
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_record(self):
        """Test updating a record that doesn't exist"""
        self.setup_pg_vector()
        pg_vector = self.pg_vector

        try:
            # Ensure table exists
            await pg_vector.ensure_table_exists()
            
            # Try to update a record that doesn't exist
            updates = {"status": "completed"}
            result = await pg_vector.update_record("nonexistent_id", updates)
            assert result is False
            
        finally:
            # await pg_vector.close()
            pass
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_record(self):
        """Test deleting a record that doesn't exist"""
        self.setup_pg_vector()
        pg_vector = self.pg_vector

        try:
            # Ensure table exists
            await pg_vector.ensure_table_exists()
            
            # Try to delete a record that doesn't exist
            result = await pg_vector.delete_record("nonexistent_id")
            assert result is False
            
        finally:
            # await pg_vector.close()
            pass


if __name__ == "__main__":
    async def run_all_tests():
        """Run all tests one by one"""
        test_instance = TestPGVector()
        time_interval = 50  # seconds
        
        await test_instance.test_init()
        sleep(time_interval)
        await test_instance.test_insert_record()
        sleep(time_interval)
        await test_instance.test_update_record()
        sleep(time_interval)
        await test_instance.test_delete_record()
        sleep(time_interval)
        await test_instance.test_get_nonexistent_record()
        sleep(time_interval)
        await test_instance.test_update_nonexistent_record()
        sleep(time_interval)
        await test_instance.test_delete_nonexistent_record()
        
        # Close PGVector session after all tests
        await test_instance.close_pg_vector()
    
    # Run all tests in a single event loop
    asyncio.run(run_all_tests())
    