import logging
import os
from datetime import datetime
from typing import Optional

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from app.utils.pg_vector.table import Base, OCRTable


class PGVector:
    engine: AsyncEngine = None
    connection_string: str = ""
    table: OCRTable = None
    engine_creation_time: datetime = None
    flush_interval_seconds: float = None

    def __init__(self, connection_string="", flush_interval_seconds: float = 30):
        if not connection_string:
            connection_string = self.get_connection_string()
        self.flush_interval_seconds = flush_interval_seconds
        self.connection_string = connection_string
        self.table = OCRTable

    async def ensure_engine(self) -> AsyncEngine:
        if not self.engine:
            self.engine = create_async_engine(
                self.connection_string, echo=False, future=True
            )
            self.engine_creation_time = datetime.utcnow()

        engine_age = (datetime.utcnow() - self.engine_creation_time).total_seconds()
        if engine_age > self.flush_interval_seconds:
            logging.info("Recreating engine due to flush interval")
            await self.engine.dispose()
            self.engine = create_async_engine(
                self.connection_string, echo=False, future=True
            )
            self.engine_creation_time = datetime.utcnow()

        return self.engine

    async def get_session(self) -> AsyncSession:
        engine = await self.ensure_engine()
        async_session = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        return async_session()

    def get_connection_string(self):
        """PostgreSQL connection string"""
        connection_string = os.getenv("POSTGRES_URL_NO_SSL_DEV")
        if connection_string:
            # Convert to async driver format
            if connection_string.startswith("postgresql://"):
                connection_string = connection_string.replace(
                    "postgresql://", "postgresql+asyncpg://", 1
                )
            return connection_string
        raise ValueError("No connection string found in environment")

    async def ensure_table_exists(self):
        """Create the OCR table if it doesn't exist"""
        try:
            engine = await self.ensure_engine()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logging.info("Successfully ensured OCR table exists")
        except Exception as e:
            logging.error(f"Error creating table: {e}")
            raise

    async def upsert_record(self, record: OCRTable):
        """Insert or update a record in the OCR table"""
        try:
            async with await self.get_session() as session:
                # Convert record to dict for upsert
                record_dict = dict(record)

                # Create upsert statement
                stmt = insert(OCRTable).values(**record_dict)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"], set_=record_dict
                )

                await session.execute(stmt)
                await session.commit()
                logging.info(f"Successfully upserted record with id: {record.id}")
                return True
        except Exception as e:
            logging.error(f"Error upserting record: {e}")
            await session.rollback()
            raise

    async def get_record_by_id(self, record_id: str) -> Optional[OCRTable]:
        """Retrieve a record by its ID"""
        try:
            async with await self.get_session() as session:
                stmt = select(OCRTable).where(OCRTable.id == record_id)
                result = await session.execute(stmt)
                record = result.scalar_one_or_none()

                if record:
                    logging.info(f"Successfully retrieved record with id: {record_id}")
                else:
                    logging.info(f"No record found with id: {record_id}")

                return record
        except Exception as e:
            logging.error(f"Error retrieving record by id {record_id}: {e}")
            raise

    async def update_record(self, record_id: str, updates: OCRTable) -> bool:
        """Update specific fields of a record"""
        try:
            async with await self.get_session() as session:
                stmt = (
                    update(OCRTable)
                    .where(OCRTable.id == record_id)
                    .values(**dict(updates))
                )

                result = await session.execute(stmt)
                await session.commit()

                if result.rowcount > 0:
                    logging.info(f"Successfully updated record with id: {record_id}")
                    return True
                else:
                    logging.warning(f"No record found to update with id: {record_id}")
                    return False
        except Exception as e:
            logging.error(f"Error updating record {record_id}: {e}")
            await session.rollback()
            raise

    async def delete_record(self, record_id: str) -> bool:
        """Delete a record by its ID"""
        try:
            async with await self.get_session() as session:
                stmt = delete(OCRTable).where(OCRTable.id == record_id)
                result = await session.execute(stmt)
                await session.commit()

                if result.rowcount > 0:
                    logging.info(f"Successfully deleted record with id: {record_id}")
                    return True
                else:
                    logging.warning(f"No record found to delete with id: {record_id}")
                    return False
        except Exception as e:
            logging.error(f"Error deleting record {record_id}: {e}")
            await session.rollback()
            raise

    async def flush(self):
        """Flush the current session"""
        try:
            async with await self.get_session() as session:
                await session.flush()
                logging.info("Successfully flushed the session")
        except Exception as e:
            logging.error(f"Error flushing session: {e}")
            raise

    async def close(self):
        """Close the database connection and session"""
        try:
            if self.engine:
                await self.engine.dispose()
            logging.info("Successfully closed database connections")
        except Exception as e:
            logging.error(f"Error closing database connections: {e}")
            raise
