from sqlalchemy import (
    Column, Integer, String, Text, Boolean, TIMESTAMP, ForeignKey, func, Table
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# --------------------------
# Association Tables
# --------------------------

place_registrar_table = Table(
    "place_registrar",
    Base.metadata,
    Column("place_id", Integer, ForeignKey("public.place.place_id", ondelete="CASCADE"), primary_key=True),
    Column("user_id", Integer, ForeignKey("public.app_user.user_id", ondelete="CASCADE"), primary_key=True),
    Column("created_at", TIMESTAMP, nullable=False, server_default=func.current_timestamp()),
    schema="public"
)

place_category_table = Table(
    "place_category",
    Base.metadata,
    Column("category_id", Integer, ForeignKey("public.category.category_id"), primary_key=True),
    Column("place_id", Integer, ForeignKey("public.place.place_id"), primary_key=True),
    schema="public"
)

# --------------------------
# Main Tables
# --------------------------

class AppUser(Base):
    __tablename__ = "app_user"
    __table_args__ = {"schema": "public"}

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(50), nullable=False)
    name = Column(String(20), nullable=False)
    status = Column(String(10), nullable=False, default="active")
    role = Column(String(10), nullable=False)
    provider = Column(String(6), nullable=True)
    provider_user_id = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.current_timestamp())
    deleted_at = Column(TIMESTAMP, nullable=True)
    last_login_at = Column(TIMESTAMP, server_default=func.current_timestamp())

    itineraries = relationship("Itinerary", back_populates="user")
    places_registered = relationship("Place", secondary=place_registrar_table, back_populates="registrars")


class Place(Base):
    __tablename__ = "place"
    __table_args__ = {"schema": "public"}

    place_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(30), nullable=False)
    address = Column(String(255), nullable=False)
    address_la = Column(Text, nullable=False)
    address_lo = Column(Text, nullable=False)
    type = Column(String(20), nullable=False)
    count = Column(Integer, nullable=False, default=0)
    website = Column(Text, nullable=True)
    image_url = Column(Text, nullable=True)
    insta_nickname = Column(Text, nullable=True)
    open_hour = Column(Text, nullable=True)
    close_hour = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    deleted_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())

    registrars = relationship("AppUser", secondary=place_registrar_table, back_populates="places_registered")
    categories = relationship("Category", secondary=place_category_table, back_populates="places")
    itinerary_items = relationship("ItineraryItem", back_populates="place")


class Accommodation(Base):
    __tablename__ = "accommodation"
    __table_args__ = {"schema": "public"}

    accommodation_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    address = Column(String(255), nullable=False)
    address_la = Column(Text, nullable=False)
    address_lo = Column(Text, nullable=False)
    type = Column(String(20), nullable=False)
    phone = Column(String(30), nullable=True)
    image_url = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    deleted_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())

    itinerary_items = relationship("ItineraryItem", back_populates="accommodation")


class Itinerary(Base):
    __tablename__ = "itinerary"
    __table_args__ = {"schema": "public"}

    itinerary_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("public.app_user.user_id"), nullable=True)
    relation = Column(String(10), nullable=True)
    start_at = Column(TIMESTAMP, nullable=False)
    end_at = Column(TIMESTAMP, nullable=False)
    start_location = Column(String(30), nullable=False)
    theme = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    deleted_at = Column(TIMESTAMP, nullable=True)
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())

    user = relationship("AppUser", back_populates="itineraries")
    items = relationship("ItineraryItem", back_populates="itinerary")


class ItineraryItem(Base):
    __tablename__ = "itinerary_item"
    __table_args__ = {"schema": "public"}

    item_id = Column(Integer, primary_key=True, autoincrement=True)
    itinerary_id = Column(Integer, ForeignKey("public.itinerary.itinerary_id"), nullable=False)
    place_id = Column(Integer, ForeignKey("public.place.place_id"), nullable=False)
    accommodation_id = Column(Integer, ForeignKey("public.accommodation.accommodation_id"), nullable=True)
    start_time = Column(TIMESTAMP, nullable=False)
    end_time = Column(TIMESTAMP, nullable=True)
    is_required = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    deleted_at = Column(TIMESTAMP, nullable=True)

    itinerary = relationship("Itinerary", back_populates="items")
    place = relationship("Place", back_populates="itinerary_items")
    accommodation = relationship("Accommodation", back_populates="itinerary_items")


class Category(Base):
    __tablename__ = "category"
    __table_args__ = {"schema": "public"}

    category_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20), nullable=False)
    status = Column(String(10), nullable=False, default="active")
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())

    places = relationship("Place", secondary=place_category_table, back_populates="categories")