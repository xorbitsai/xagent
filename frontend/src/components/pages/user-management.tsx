"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog";
import { Trash2, Users, Search, ChevronLeft, ChevronRight, ArrowLeft } from "lucide-react";
import { useAuth } from "@/contexts/auth-context";
import { apiRequest } from "@/lib/api-wrapper";
import { getApiUrl } from "@/lib/utils";
import Link from "next/link";
import { useI18n } from "@/contexts/i18n-context";

interface User {
  id: number;
  username: string;
  is_admin: boolean;
  created_at: string;
  updated_at: string;
}

interface UserListResponse {
  users: User[];
  total: number;
  page: number;
  size: number;
  pages: number;
}

export default function UserManagement() {
  const { user } = useAuth();
  const { t, locale } = useI18n();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalUsers, setTotalUsers] = useState(0);
  const [pageSize] = useState(20);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams({
        page: currentPage.toString(),
        size: pageSize.toString(),
      });

      if (searchTerm) {
        params.append("search", searchTerm);
      }

      const response = await apiRequest(`${getApiUrl()}/api/admin/users?${params}`);

      if (!response.ok) {
        if (response.status === 403) {
          return;
        }
        const errorText = await response.text();
        throw new Error(errorText || "Failed to fetch users");
      }

      const data: UserListResponse = await response.json();
      setUsers(data.users);
      setTotalPages(data.pages);
      setTotalUsers(data.total);
    } catch (error: any) {
      console.error("Error fetching users:", error);
    } finally {
      setLoading(false);
    }
  };

  const deleteUser = async (userId: number, username: string) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/admin/users/${userId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to delete user");
      }

      alert(`${t('userManagement.list.alerts.delete_success_prefix')}${username}${t('userManagement.list.alerts.delete_success_suffix')}`);

      // Refresh users list
      fetchUsers();
    } catch (error: any) {
      console.error("Error deleting user:", error);
      alert(`${t('userManagement.list.alerts.delete_failed_prefix')}${error.message || t('userManagement.list.alerts.delete_failed_suffix')}`);
    }
  };

  useEffect(() => {
    if (user?.is_admin) {
      fetchUsers();
    }
  }, [currentPage, searchTerm, user]);

  // Redirect if not admin and user is loaded
  if (user && !user.is_admin) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-2">{t('userManagement.list.no_permission.title')}</h2>
          <p className="text-muted-foreground">{t('userManagement.list.no_permission.description')}</p>
        </div>
      </div>
    );
  }

  // Show loading if user is not loaded yet
  if (!user) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <h2 className="text-2xl font-semibold mb-2">{t('userManagement.list.loading.title')}</h2>
          <p className="text-muted-foreground">{t('userManagement.list.loading.description')}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto bg-[#0E1117]">
      <div className="p-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-[22px] font-bold leading-tight">{t('userManagement.title')}</h1>
            <p className="text-muted-foreground">
              {t('userManagement.description')}
            </p>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="h-4 w-4" />
            {t('userManagement.stats.total_users_prefix')}{totalUsers}{t('userManagement.stats.total_users_suffix')}
          </div>
        </div>

        <Card className="bg-card/50 border-border">
        <CardHeader>
          <CardTitle>{t('userManagement.list.title')}</CardTitle>
          <CardDescription>
            {t('userManagement.list.description')}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4 mb-6">
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder={t('userManagement.list.search_placeholder')}
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
              <p className="mt-2 text-muted-foreground">{t('userManagement.list.loading.title')}</p>
            </div>
          ) : (
            <>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>{t('userManagement.list.table.id')}</TableHead>
                      <TableHead>{t('userManagement.list.table.username')}</TableHead>
                      <TableHead>{t('userManagement.list.table.role')}</TableHead>
                      <TableHead>{t('userManagement.list.table.created_at')}</TableHead>
                      <TableHead>{t('userManagement.list.table.updated_at')}</TableHead>
                      <TableHead className="text-right">{t('userManagement.list.table.actions')}</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {users.map((userItem) => (
                      <TableRow key={userItem.id}>
                        <TableCell className="font-medium">
                          {userItem.id}
                        </TableCell>
                        <TableCell>{userItem.username}</TableCell>
                        <TableCell>
                          {userItem.is_admin ? (
                            <Badge variant="default">{t('userManagement.list.table.admin')}</Badge>
                          ) : (
                            <Badge variant="secondary">{t('userManagement.list.table.normal')}</Badge>
                          )}
                        </TableCell>
                        <TableCell>
                          {new Date(userItem.created_at).toLocaleString(locale)}
                        </TableCell>
                        <TableCell>
                          {new Date(userItem.updated_at).toLocaleString(locale)}
                        </TableCell>
                        <TableCell className="text-right">
                          {userItem.id !== parseInt(user?.id || "0") && (
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  className="text-destructive hover:text-destructive"
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>{t('userManagement.list.table.delete_confirm_title')}</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    {`${t('userManagement.list.table.delete_confirm_description_prefix')}${userItem.username}${t('userManagement.list.table.delete_confirm_description_suffix')}`}
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>{t('userManagement.list.table.cancel')}</AlertDialogCancel>
                                  <AlertDialogAction
                                    onClick={() => deleteUser(userItem.id, userItem.username)}
                                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                  >
                                    {t('userManagement.list.table.confirm_delete')}
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>

              {totalPages > 1 && (
                <div className="flex items-center justify-between mt-6">
                  <div className="text-sm text-muted-foreground">
                    {t('userManagement.list.pagination.summary', {
                      from: (currentPage - 1) * pageSize + 1,
                      to: Math.min(currentPage * pageSize, totalUsers),
                      total: totalUsers,
                    })}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(currentPage - 1)}
                      disabled={currentPage <= 1}
                    >
                      <ChevronLeft className="h-4 w-4" />
                      {t('userManagement.list.pagination.prev')}
                    </Button>
                    <div className="text-sm text-muted-foreground">
                      {t('userManagement.list.pagination.page', { current: currentPage, total: totalPages })}
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setCurrentPage(currentPage + 1)}
                      disabled={currentPage >= totalPages}
                    >
                      {t('userManagement.list.pagination.next')}
                      <ChevronRight className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
